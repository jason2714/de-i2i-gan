import importlib


def find_trainer_using_model_name(model_name):
    # Given the option --model [modelname],
    # the file "trainers/modelname_trainer.py"
    # will be imported.
    trainer_filename = "trainers." + model_name + "_trainer"
    trainerlib = importlib.import_module(trainer_filename)

    # In the file, the class called ModelNameTrainer() will
    # be instantiated, and it is case-insensitive.
    trainer = None
    target_trainer_name = model_name.replace('_', '') + 'trainer'
    for name, cls in trainerlib.__dict__.items():
        if name.lower() == target_trainer_name.lower():
            trainer = cls

    if trainer is None:
        print(
            "In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (
                trainer_filename, target_trainer_name))
        exit(0)

    return trainer
