# encoding: utf-8
import os


def check_model_path(model_path):
    if not os.path.exists(model_path):
        raise ValueError(
            "Model path does not exist: {}, please download and move to 'models/'".format(
                model_path
            )
        )
