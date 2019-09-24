azerty_shift = [
 '²1234567890°+',
 ' AZERTYUIOP¨£',
 ' QSDFGHJKLM% ',
 '>WXCVBN?./§  ']

azerty_noshift = [
 '²&é"\'(-è_çà)=',
 ' azertyuiop^$',
 ' qsdfghjklmù ',
 '<wxcvbn,;:!  ']

azerty_alt = '¹ˇ~#{[|`\\^@]}'

char_list = '²1234567890°+ AZERTYUIOP¨£QSDFGHJKLM%>WXCVBN?./§&é"\'(-è_çà)=azertyuiop^$qsdfghjklmù<wxcvbn,;:!¹ˇ~#{[|`\\^@]}'

def find_pos(key):
    if key in '²1234567890°+ AZERTYUIOP¨£QSDFGHJKLM%>WXCVBN?./§':
        for x, row in enumerate(azerty_shift):
            for y, col in enumerate(row):
                if col == key:
                    return x, y
    elif key in '&é"\'(-è_çà)=azertyuiop^$qsdfghjklmù<wxcvbn,;:!':
        for x, row in enumerate(azerty_noshift):
            for y, col in enumerate(row):
                if col == key:
                    return x, y
    elif key in azerty_alt:
        x = 0
        for y, col in enumerate(azerty_alt):
            if col == key:
                return x, y
    return -1, -1

def key_dist(key1, key2):
    x1, y1 = find_pos(key1)
    x2, y2 = find_pos(key2)
    # print(x1, y1)
    if x1 == x2:
        return abs(y2 - y1)
    elif y1 == y2:
        return abs(y2 - y1)
    return max(abs(x2 - x1),abs(y2 - y1))

def build_azerty():
    map_dict = {k:{t:(key_dist(t,k)) for t in char_list} for k in char_list}
    return map_dict

key_dist = build_azerty()
