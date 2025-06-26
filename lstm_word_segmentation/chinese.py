from icu import Char, UProperty, UnicodeString

radical_number = Char.getIntPropertyValue(0x51D2, Char.RADICAL)
print(radical_number)