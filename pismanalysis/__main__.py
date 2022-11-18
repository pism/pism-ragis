import argparse
import errno
import os

from configobj import ConfigObj
from validate import Validator


def _validate_config(config):
    validator = Validator()

    result = config.validate(validator, preserve_errors=True)
    if result is not True:
        print("Validation results={}".format(result))
        raise ValueError(
            "Configuration file validation failed. Parameters listed as false in the validation results have invalid values."
        )
    else:
        print("Configuration file validation passed.")


# FIXME: C901 'main' is too complex (19)
def main():  # noqa: C901

    # Command line parser
    parser = argparse.ArgumentParser(
        description="Launch PISM post-processing tasks", usage="pypac -c <config>"
    )
    parser.add_argument(
        "-c", "--config", type=str, help="configuration file", required=True
    )
    args = parser.parse_args()

    # Subdirectory where templates are located
    templateDir = os.path.join(os.path.dirname(__file__), "templates")

    # Read configuration file and validate it
    default_config = os.path.join(templateDir, "default.ini")
    user_config = ConfigObj(args.config, configspec=default_config)
    if "campaign" in user_config["default"]:
        campaign = user_config["default"]["campaign"]
    else:
        campaign = "none"
    if campaign != "none":
        campaign_file = os.path.join(templateDir, "{}.cfg".format(campaign))
        if not os.path.exists(campaign_file):
            raise ValueError(
                "{} does not appear to be a known campaign".format(campaign)
            )
        campaign_config = ConfigObj(campaign_file, configspec=default_config)
        # merge such that user_config takes priority over campaign_config
        campaign_config.merge(user_config)
        config = campaign_config
    else:
        config = user_config
    _validate_config(config)

    # Add templateDir to config
    config["default"]["templateDir"] = templateDir

    # Output script directory
    output = config["default"]["output"]
    username = os.environ.get("USER")
    output = output.replace("$USER", username)
    scriptDir = os.path.join(output, "post/scripts")
    try:
        os.makedirs(scriptDir)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise OSError("Cannot create script directory")
        pass

    # if ("machine" not in config["default"]) or (config["default"]["machine"] == ""):
    #     # MachineInfo below will then call `discover_machine()`,
    #     # which only works on log-in nodes.
    #     machine = None
    # else:
    #     # If `machine` is set, then MachineInfo can bypass the
    #     # `discover_machine()` function.
    #     machine = config["default"]["machine"]
    # machine_info = MachineInfo(machine=machine)


if __name__ == "__main__":
    main()
