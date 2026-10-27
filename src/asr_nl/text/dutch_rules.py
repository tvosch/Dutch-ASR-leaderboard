"""Dutch rule tables for the v2 normalizer: contractions, units and year context words.

The contraction rules are from the Dutch rule file (normalizers/european/nl.json) of
https://github.com/ThomasKluiters/european-normalizer at commit 6c9ea6c.

    MIT License

    Copyright (c) 2023 Kurian Benoy

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

Patterns are regular expressions applied to lowercased text with apostrophes
normalized to "'". Longer patterns are tried first, so "z'n" wins over "'n".
"""

CONTRACTIONS = {
    r"\bzo'n\b": "zo een",
    r"\bz'n\b": "zijn",
    r"\bm'n\b": "mijn",
    r"\bd'r\b": "er",
    r"'t\b": " het",
    r"'n\b": " een",
    r"'k\b": " ik",
    r"'m\b": " hem",
    r"'r\b": " haar",
    r"(?<![a-z])'s\b": " des",
    r"(?<=[a-z])'s\b": "s",
}


# Unit abbreviations, spelled out when they directly follow a number ("35mm",
# "10 km/u"), so they match references that write the unit in full. Only after
# a number: a bare "m" or "g" elsewhere is left alone. "²"/"³" are already "2"/"3"
# after NFKC.
UNITS = {
    # length
    "mm": "millimeter", "cm": "centimeter", "dm": "decimeter", "m": "meter", "km": "kilometer",
    # area and volume
    "mm2": "vierkante millimeter", "cm2": "vierkante centimeter", "m2": "vierkante meter",
    "km2": "vierkante kilometer", "ha": "hectare", "cm3": "kubieke centimeter", "m3": "kubieke meter",
    # mass
    "mg": "milligram", "gr": "gram", "kg": "kilogram",  # no bare "g": 2G/4G/5G are network generations
    # liquid volume
    "ml": "milliliter", "cl": "centiliter", "dl": "deciliter", "l": "liter",
    # speed
    "km/u": "kilometer per uur", "km/h": "kilometer per uur", "km/uur": "kilometer per uur",
    "m/s": "meter per seconde",
    # time
    "ms": "milliseconde", "sec": "seconden", "min": "minuten", "u": "uur",
    # temperature ("°" alone is already "graden")
    "°c": "graden celsius", "°f": "graden fahrenheit",
    # data
    "kb": "kilobyte", "mb": "megabyte", "gb": "gigabyte", "tb": "terabyte",
    # frequency and energy
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "kw": "kilowatt", "kwh": "kilowattuur", "mw": "megawatt", "gw": "gigawatt",
}

# Words before a two-part spoken year that mark it as a year rather than a
# clock time: "in negentien vijfenveertig" -> 1945, "om negentien vijfenveertig" stays a time.
YEAR_CONTEXT = {"in", "sinds", "jaar", "anno", "van", "tot", "na", "voor", "rond", "omstreeks", "begin", "eind", "midden"}
