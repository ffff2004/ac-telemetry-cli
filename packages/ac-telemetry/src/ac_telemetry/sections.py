import configparser
import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SectionIni:
    section_id: str
    name: str
    in_progress: float
    out_progress: float | None


def parse_sections_ini(sections_text: str) -> tuple[SectionIni, ...]:
    """Parse ``SECTION_*`` entries from Assetto Corsa ``sections.ini`` text."""
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.optionxform = str
    try:
        parser.read_string(sections_text)
    except configparser.Error as exc:
        raise ValueError(f"Invalid sections.ini: {exc}") from exc

    sections: list[SectionIni] = []
    for section_id in parser.sections():
        if not section_id.startswith("SECTION_"):
            continue
        values = parser[section_id]
        try:
            in_progress = float(values["IN"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Section {section_id!r} has an invalid IN value") from exc
        if not math.isfinite(in_progress) or not 0.0 <= in_progress <= 1.0:
            raise ValueError(f"Section {section_id!r} IN must be between 0.0 and 1.0")

        out_text = values.get("OUT")
        if out_text is None or not out_text.strip():
            out_progress = None
        else:
            try:
                out_progress = float(out_text)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Section {section_id!r} has an invalid OUT value"
                ) from exc
            if not math.isfinite(out_progress) or not 0.0 <= out_progress <= 1.0:
                raise ValueError(
                    f"Section {section_id!r} OUT must be between 0.0 and 1.0"
                )

        sections.append(
            SectionIni(
                section_id=section_id,
                name=values.get("TEXT", section_id) or section_id,
                in_progress=in_progress,
                out_progress=out_progress,
            )
        )
    return tuple(sections)
