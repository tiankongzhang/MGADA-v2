# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch

from fcos_core.utils.model_serialization import load_state_dict
from fcos_core.utils.c2_model_loading import load_c2_format
from fcos_core.utils.imports import import_file
from fcos_core.utils.model_zoo import cache_url


class Checkpointer(object):
    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_checkpoint_file(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read()
                last_saved = last_saved.strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            last_saved = ""
        return last_saved

    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint):
        load_state_dict(self.model, checkpoint.pop("model"))


class DetectronCheckpointer(Checkpointer):
    def __init__(
        self,
        cfg,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        super(DetectronCheckpointer, self).__init__(
            model, optimizer, scheduler, save_dir, save_to_disk, logger
        )
        self.cfg = cfg.clone()

    def _load_file(self, f):
        # catalog lookup
        if f.startswith("catalog://"):
            paths_catalog = import_file(
                "maskrcnn_benchmark.config.paths_catalog", self.cfg.PATHS_CATALOG, True
            )
            catalog_f = paths_catalog.ModelCatalog.get(f[len("catalog://") :])
            self.logger.info("{} points to {}".format(f, catalog_f))
            f = catalog_f
        # download url files
        if f.startswith("http"):
            # if the file is a url path, download it and cache it
            cached_f = cache_url(f)
            self.logger.info("url {} cached in {}".format(f, cached_f))
            f = cached_f
        # convert Caffe2 checkpoint from pkl
        if f.endswith(".pkl"):
            return load_c2_format(self.cfg, f)
        # load native detectron.pytorch checkpoint
        loaded = super(DetectronCheckpointer, self)._load_file(f)
        if "model" not in loaded and "model_backbone" not in loaded:
            loaded = dict(model=loaded)
        return loaded

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model_tc_backbone"] = self.model.teacher_backbone.state_dict()
        data["model_tc_fcos"] = self.model.teacher_fcos.state_dict()
        
        data["model_st_backbone"] = self.model.student_backbone.state_dict()
        data["model_st_fcos"] = self.model.student_fcos.state_dict()

        if self.optimizer is not None:
                if 'student' in self.optimizer.keys():
                    data["optimizer_student_backbone"] = self.optimizer["student_backbone"].state_dict()
                    data["optimizer_student_fcos"] = self.optimizer["student_fcos"].state_dict()
                
                if 'teacher' in self.optimizer.keys():
                    data["optimizer_teacher_backbone"] = self.optimizer["teacher_backbone"].state_dict()
                    data["optimizer_teacher_fcos"] = self.optimizer["teacher_fcos"].state_dict()

        if self.scheduler is not None:
            if 'student' in self.scheduler.keys():
                data["scheduler_student_backbone"] = self.scheduler["student_backbone"].state_dict()
                data["scheduler_student_fcos"] = self.scheduler["student_fcos"].state_dict()
            
            if 'teacher' in self.scheduler.keys():
                data["scheduler_teacher_backbone"] = self.scheduler["teacher_backbone"].state_dict()
                data["scheduler_teacher_fcos"] = self.scheduler["teacher_fcos"].state_dict()
                

        # data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        self.tag_last_checkpoint(save_file)

    def load(self, f=None, load_dis=True, load_opt_sch=False):
        if self.has_checkpoint():
            # override argument with existing checkpoint
            f = self.get_checkpoint_file()
        if not f:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        
        checkpoint = self._load_file(f)
        self._load_model(checkpoint, load_dis)

        if load_opt_sch:
            if "optimizer_" in checkpoint and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(f))

                self.optimizer["student_backbone"].load_state_dict(checkpoint.pop("optimizer_student_backbone"))
                self.optimizer["student_fcos"].load_state_dict(checkpoint.pop("optimizer_student_fcos"))
                self.optimizer["teacher_backbone"].load_state_dict(checkpoint.pop("optimizer_teacher_backbone"))
                self.optimizer["teacher_fcos"].load_state_dict(checkpoint.pop("optimizer_teacher_fcos"))

            else:
                self.logger.info(
                    "No optimizer found in the checkpoint. Initializing model from scratch"
                )

            if "scheduler_" in checkpoint and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(f))
                
                self.scheduler["student_backbone"].load_state_dict(checkpoint.pop("scheduler_student_backbone"))
                self.scheduler["student_fcos"].load_state_dict(checkpoint.pop("scheduler_student_fcos"))
                self.scheduler["teacher_backbone"].load_state_dict(checkpoint.pop("scheduler_teacher_backbone"))
                self.scheduler["teacher_fcos"].load_state_dict(checkpoint.pop("scheduler_teacher_fcos"))

            else:
                self.logger.info(
                    "No scheduler found in the checkpoint. Initializing model from scratch"
                )

        # return any further checkpoint data
        return checkpoint

    def _load_model(self, checkpoint, load_dis=True):
        if "model_student" in checkpoint:
            # load checkpoint of our model
            load_state_dict(self.model.student_backbone, checkpoint.pop("model_st_backbone"))
            load_state_dict(self.model.student_fcos, checkpoint.pop("model_st_fcos"))
        
        if "model_teacher" in checkpoint:
            # load checkpoint of our model
            load_state_dict(self.model.teacher_backbone, checkpoint.pop("model_tc_backbone"))
            load_state_dict(self.model.teacher_fcos, checkpoint.pop("model_tc_fcos"))
