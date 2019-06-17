from collections import OrderedDict


def rename_dict(dic, new_names):
    '''new_names is a list of new names'''
    new_dic = OrderedDict()
    assert len(dic) == len(new_names), "the lengths of old keys and new keys are different!"

    length = len(dic)

    for i in range(length):
        k, v = dic.popitem(False)
        new_dic[new_names[i]] = v

    return new_dic


def get_new_names(old_names, start_pos, offset):
    '''start_pos is a list, and should contain -inf, + inf as the first and last elements, respectively'''
    new_names = list()

    left, right = start_pos[0], start_pos[1]
    current_offset = offset[0]

    pos_idx = 0
    offset_idx = 0

    for name in old_names:
        name_split = name.split('.')

        number = int(name_split[1])

        if number >= right:
            pos_idx += 1
            offset_idx += 1

            left, right = start_pos[pos_idx], start_pos[pos_idx + 1]
            current_offset = offset[offset_idx]

        number += current_offset

        name_split[1] = str(number)
        new_name = '.'.join(name_split)

        new_names.append(new_name)

    return new_names

