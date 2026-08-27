from ac_telemetry.sections import SectionIni, parse_sections_ini


def test_parse_sections_ini_returns_normalized_sections_in_file_order() -> None:
    sections = parse_sections_ini(
        """
[SECTION_1]
IN=0.6
OUT=0.8
TEXT=Second Corner

[IGNORED]
IN=0.1

[SECTION_0]
IN=0.2
OUT=0.4
TEXT=First Corner
"""
    )

    assert sections == (
        SectionIni(
            section_id="SECTION_1",
            name="Second Corner",
            in_progress=0.6,
            out_progress=0.8,
        ),
        SectionIni(
            section_id="SECTION_0",
            name="First Corner",
            in_progress=0.2,
            out_progress=0.4,
        ),
    )


def test_parse_sections_ini_allows_missing_out_for_consumers_that_do_not_need_it() -> (
    None
):
    sections = parse_sections_ini("[SECTION_0]\nIN=0.2\n")

    assert sections == (
        SectionIni(
            section_id="SECTION_0",
            name="SECTION_0",
            in_progress=0.2,
            out_progress=None,
        ),
    )
