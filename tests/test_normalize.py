"""Tests for the Dutch text normalizer (no GPU, network or model downloads needed)."""

import pytest

from asr_nl.text import NORMALIZER_VERSION, normalize_text

# Reference style (lowercase, no punctuation, often spelled-out numbers, as in
# FLEURS/VoxPopuli/MLS) paired with the way a model typically writes the same
# words. Both must normalize to the same string, so formatting is not counted
# as a word error.
EQUIVALENCES = {
    "punctuation_and_case": ("de kat zit op de mat", "De kat zit op de mat."),
    "curly_apostrophe": ("zo'n kat", "zo’n kat"),
    "digits_on_both_sides": ("op 15 augustus 1940", "Op 15 augustus 1940."),
    "spelled_vs_digits": ("op vijftien augustus", "op 15 augustus"),
    "spelled_vs_digits_year": ("in tweeduizendveertien", "in 2014"),
    "thousands_separator": ("tienduizend jaar", "10.000 jaar"),
    "large_number_mixed": ("vijf miljard euro", "5 miljard euro"),
    "number_one_accent": ("één miljard euro", "1 miljard euro"),
    "decimal_comma": ("drie komma vijf graden", "3,5 graden"),
    "decimal_point": ("drie punt vijf", "3.5"),
    "percent": ("twintig procent", "20%"),
    "ordinal_e": ("de derde eeuw", "de 3e eeuw"),
    "ordinal_ste": ("de eenentwintigste eeuw", "de 21ste eeuw"),
    "score_dash": ("het werd zes zes", "het werd 6-6"),
    "ampersand": ("larson en lafasto", "Larson & Lafasto"),
    "euro_sign": ("twintig euro", "€20"),
    "currency_scale": ("drie komma vijf miljoen euro", "€3,5 miljoen"),
    "dollar": ("vijf dollar", "$5"),
    "hyphen_joined": ("email", "e-mail"),
    "stress_accent": ("dat is te echt", "Dat is té echt"),
    "filler": ("ja dat klopt", "Uh, ja dat klopt."),
    "contraction_zn_mn": ("z'n vader en m'n moeder", "zijn vader en mijn moeder"),
    "contraction_t": ("'t is laat", "het is laat"),
    "spoken_parenthetical": ("zo deel je het in drieën zie voorbeeld", "Zo deel je het in drieën (zie voorbeeld)."),
    "nested_parenthesis": ("de middeleeuwen de europese geschiedenis in de vroege eeuw",
                           "De middeleeuwen (de Europese geschiedenis (in de vroege eeuw))."),
    "units_mm": ("vijfendertig millimeter", "35mm"),
    "units_kmh": ("honderdtwintig kilometer per uur", "120 km/u"),
    "units_m2": ("tien vierkante meter", "10 m²"),
    "celsius": ("dertig graden celsius", "30°C"),
    "year_in_parts": ("in achttien drieëntachtig", "in 1883"),
    "year_context": ("in negentien vijfenveertig", "in 1945"),
    "time_colon": ("om tien uur eenendertig", "om 10:31"),
    "time_dot_uur": ("om tien uur eenendertig", "om 10.31 uur"),
    "time_whole": ("om tweeëntwintig uur", "om 22:00"),
    "old_spelling_number": ("achtentwintig", "acht-en-twintig"),
    "separate_numbers": ("van vijf en honderd dollar", "van 5 en 100 dollar"),
    "network_generation": ("de twee g technologie", "de 2G technologie"),
    "duration_uur": ("drie uur rustig", "3 uur rustig"),
    "glued_digit": ("het m zestien geweer", "het M16-geweer"),
    "jaren_80": ("de jaren tachtig", "de jaren 80"),
    "op_een_na": ("op één na", "op 1 na"),
    "elided_hyphen": ("drie en vierlettergrepige woorden", "drie- en vierlettergrepige woorden"),
    # Medical text (HoMed Medicijnjournaal)
    "drug_class_digit": ("glp één analogen", "GLP-1-analogen"),
    "drug_class_glued": ("dpp vier remmer", "DPP4-remmer"),
    "dose": ("tien milligram", "10 mg"),
    "dose_decimal": ("twee komma vijf milligram", "2,5 mg"),
    "diabetes_type": ("type twee diabetes", "type 2 diabetes"),
    "one_in_ten": ("een op de tien", "1 op de 10"),
}

# Known limitations: these still score as an error. strict=True makes the test
# fail if one starts passing, so this list stays accurate.
KNOWN_LIMITATIONS = {
    # Hyphen joins, a space does not: "noord holland" is a spelling error in Dutch.
    "hyphen_split": ("noord holland", "Noord-Holland"),
    # Abbreviations: rare in model outputs and an open-ended list.
    "abbreviation": ("de heer jansen", "Dhr. Jansen"),
    "abbreviation_dots": ("het us corps", "het U.S. Corps"),
    "abbr_enz": ("aap noot mies enzovoort", "aap noot mies enz."),
    "abbr_bijv": ("bijvoorbeeld", "bijv."),
    "abbr_oa": ("onder andere", "o.a."),
    "abbr_nr": ("nummer vijf", "nr. 5"),
    "abbr_mevr": ("mevrouw jansen", "mevr. Jansen"),
    "abbr_ca": ("circa tien", "ca. 10"),
    # One-word spoken years: text2num does not parse them; rare in references and outputs.
    "spoken_year_one_word": ("in negentienveertig", "in 1940"),
    # Clock times without "uur", and "half elf": ambiguous without context.
    "spoken_time_no_uur": ("om tien eenendertig", "om 10:31"),
    "half_hour": ("om half elf", "om 10:30"),
    "fraction": ("negenentwintig driekwart inch", "29¾ inch"),
}


def test_version():
    assert NORMALIZER_VERSION == 2


@pytest.mark.parametrize("ref, hyp", EQUIVALENCES.values(), ids=EQUIVALENCES.keys())
def test_reference_and_model_style_agree(ref, hyp):
    assert normalize_text(ref, "nl") == normalize_text(hyp, "nl")


@pytest.mark.parametrize("ref, hyp", [
    pytest.param(ref, hyp, id=name, marks=pytest.mark.xfail(strict=True, reason="known limitation"))
    for name, (ref, hyp) in KNOWN_LIMITATIONS.items()
])
def test_known_limitations(ref, hyp):
    assert normalize_text(ref, "nl") == normalize_text(hyp, "nl")


@pytest.mark.parametrize("raw, expected", [
    ("Zeker 20% van al het water.", "zeker 20 procent van al het water"),
    ("de auto's", "de autos"),
    ("ING en Mr. Holmes", "ing en mr holmes"),
    ("Noord-Holland, uh, ja", "noordholland ja"),
    ("Hij zei: \"nee\" — en ging.", "hij zei nee en ging"),
    ("3,5 en 3.5", "3 komma 5 en 3 punt 5"),
    ("om 10:31 en om 22:00", "om 10 uur 31 en om 22 uur"),
    ("€3,5 miljoen", "3 komma 5 miljoen euro"),
    ("in negentien zevenentachtig", "in 1987"),
    ("om negentien vijfenveertig", "om 19 45"),  # clock time, not a year
    ("de 2G en 5G netwerken", "de 2 g en 5 g netwerken"),  # no bare "g" unit
    ("een kat", "een kat"),
    ("", ""),
])
def test_canonical_form(raw, expected):
    assert normalize_text(raw, "nl") == expected


@pytest.mark.parametrize("text", [
    "Een collectie drie- en vierlettergrepige woorden.",  # elided hyphen
    "En als honderdéen uit is, dan is alles voorbij.",    # stress accent inside a number
    "Ze trekken zich vanaf de kust 80 km/l uit.",          # unit followed by a slash
    "Op 15 augustus 1940 om 10.31 uur, €3,5 miljoen en 35mm.",
    "De HbA1c-verlager en GLP-1-analogen, 2,5 mg.",
    "Zo'n 20% van de 21ste-eeuwse M16-geweren.",
])
def test_idempotent(text):
    once = normalize_text(text, "nl")
    assert normalize_text(once, "nl") == once


@pytest.mark.parametrize("raw, expected", [
    ("Zeker 20% van al het water.", "zeker twintig van al het water"),
    ("de auto's", "de autos"),
    ("Noord-Holland, uh, ja", "noordholland ja"),
    ("op 15 augustus", "op vijftien augustus"),
])
def test_version_1_unchanged(raw, expected):
    assert normalize_text(raw, "nl", version=1) == expected
