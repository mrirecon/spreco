from spreco.common import utils
import sys

file1 = sys.argv[1]
file2 = sys.argv[2]

def diff(file1, file2):

    config1 = utils.load_config(file1)
    config2 = utils.load_config(file2)

    ks1 = config1.keys()
    ks2 = config2.keys()

    common_same = {}
    common_diff = {}
    config1_has = {}
    config2_has = config2

    for k in ks1:

        if k in ks2:
            if config1[k] == config2[k]:
                common_same[k] = config1[k]
                
            else:
                common_diff["file1/"+k] = config1[k]
                common_diff["file2/"+k] = config2[k]

            config2_has.pop(k)
        else:
            config1_has[k] = config1[k]

    print("The same common-configs are ", common_same, "\n")
    print("The different common-configs are ", common_diff, "\n")

    print("The file1 has ", config1_has, "\n")
    print("The file2 has ", config2_has, "\n")

diff(file1, file2)