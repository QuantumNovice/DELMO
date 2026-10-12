"""Class correction in an IFC file (IFC2X3, IFC4, IFC4X3) with plain ``ifcopenshell``.

* :func:`reassign_class` — change an element's entity class in place of the old one: every attribute the
  new class shares is copied (GlobalId, name, placement, representation, …) and every reference to the
  old entity (containment, property sets, materials, openings, connections, types …) is pointed at the new
  one, so nothing is lost.
* :func:`add_pset` — attach a property set with simple values (str/float/int/bool).
* :func:`reclass_file` — apply a batch of changes and write a changelog.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Iterable

import ifcopenshell
import ifcopenshell.guid
import ifcopenshell.util.element as uel

import logging

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ schema helpers
def _schema(f):
    return ifcopenshell.ifcopenshell_wrapper.schema_by_name(f.schema)


def entity_class(f, label: str) -> str:
    """Schema-specific class for a portable label (``IfcWall`` works everywhere; checks it exists and is a
    product)."""
    try:
        decl = _schema(f).declaration_by_name(label)
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"{label} is not an entity in {f.schema}") from e
    if decl.is_abstract():
        raise ValueError(f"{label} is abstract in {f.schema}")
    d = decl
    while d is not None and d.name() != "IfcProduct":
        d = d.supertype()
    if d is None:
        raise ValueError(f"{label} is not an IfcProduct subtype")
    return decl.name()


def _owner_history(f):
    oh = f.by_type("IfcOwnerHistory")
    return oh[0] if oh else None


def _value(f, v):
    if isinstance(v, bool):
        return f.create_entity("IfcBoolean", v)
    if isinstance(v, int):
        return f.create_entity("IfcInteger", v)
    if isinstance(v, float):
        return f.create_entity("IfcReal", v)
    return f.create_entity("IfcLabel", str(v)[:255])


def add_pset(f, elements, name: str, props: dict) -> object:
    """New ``IfcPropertySet`` ``name`` with ``props`` attached to ``elements`` (one entity or a list)."""
    elements = elements if isinstance(elements, (list, tuple)) else [elements]
    vals = [f.create_entity("IfcPropertySingleValue", Name=k, NominalValue=_value(f, v))
            for k, v in props.items() if v is not None]
    pset = f.create_entity("IfcPropertySet", GlobalId=ifcopenshell.guid.new(), OwnerHistory=_owner_history(f),
                           Name=name, HasProperties=vals)
    f.create_entity("IfcRelDefinesByProperties", GlobalId=ifcopenshell.guid.new(), OwnerHistory=_owner_history(f),
                    RelatedObjects=list(elements), RelatingPropertyDefinition=pset)
    return pset


# ------------------------------------------------------------------ class reassignment
def reassign_class(f, element, new_class: str, predefined_type: str | None = None, type_policy: str = "keep"):
    """Replace ``element`` with an entity of ``new_class`` carrying the same identity and relationships.

    ``type_policy`` for an assigned type object of the wrong kind (e.g. an IfcWallType on what is now a
    column): ``keep`` leaves it (valid IFC, flagged in the log), ``detach`` removes this element from it.
    Returns the new entity.
    """
    new_class = entity_class(f, new_class)
    if element.is_a() == new_class:
        return element
    new = f.create_entity(new_class)
    new_attrs = {a.name() for a in _schema(f).declaration_by_name(new_class).all_attributes()}
    info = element.get_info(include_identifier=False, recursive=False)
    for k, v in info.items():
        if k in ("type", "PredefinedType") or k not in new_attrs or v is None:
            continue
        try:
            setattr(new, k, v)
        except Exception:  # noqa: BLE001 - attribute type differs between classes
            log.debug("attribute %s not copied to %s", k, new_class)
    if "PredefinedType" in new_attrs:
        for pt in (predefined_type, info.get("PredefinedType"), "NOTDEFINED"):
            if pt:
                try:
                    new.PredefinedType = pt
                    break
                except Exception:  # noqa: BLE001 - enum value not valid for the new class
                    continue
    for inv in f.get_inverse(element):
        uel.replace_attribute(inv, element, new)
    if type_policy == "detach":
        for rel in list(getattr(new, "IsTypedBy", None) or getattr(new, "IsDefinedBy", []) or []):
            if rel.is_a("IfcRelDefinesByType"):
                rest = [o for o in rel.RelatedObjects if o != new]
                if rest:
                    rel.RelatedObjects = rest
                else:
                    f.remove(rel)
    f.remove(element)
    return new


def find_element(f, ident: str | int):
    """By GlobalId or STEP id (``#123`` or ``123``)."""
    s = str(ident).strip()
    if s.lstrip("#").isdigit():
        return f.by_id(int(s.lstrip("#")))
    return f.by_guid(s)


def reclass_file(src: str | Path, dst: str | Path, changes: Iterable[tuple[str, str]], *,
                 type_policy: str = "keep", record: bool = True, changelog: str | Path | None = None) -> list[dict]:
    """Apply ``(element id, new class)`` changes to ``src`` and write ``dst``.

    ``record`` adds a ``DELMO_Reclass`` pset (previous class, time) to each changed element so the
    edit is traceable inside the IFC; ``changelog`` writes the same as CSV. Returns the log rows.
    """
    f = ifcopenshell.open(str(src))
    rows = []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for ident, new_class in changes:
        try:
            el = find_element(f, ident)
            old = el.is_a()
            gid, name = el.GlobalId, el.Name
            typed = [r for r in (getattr(el, "IsTypedBy", None) or getattr(el, "IsDefinedBy", []) or [])
                     if r.is_a("IfcRelDefinesByType")]
            new = reassign_class(f, el, new_class, type_policy=type_policy)
            if record and new.is_a() != old:
                add_pset(f, new, "DELMO_Reclass", {"PreviousClass": old, "ChangedAt": stamp})
            note = ""
            if typed and type_policy == "keep":
                note = f"still typed by {typed[0].RelatingType.is_a()} ({typed[0].RelatingType.Name})"
            rows.append({"id": ident, "global_id": gid, "name": name, "old_class": old, "new_class": new.is_a(),
                         "status": "ok" if new.is_a() != old else "unchanged", "note": note})
        except Exception as e:  # noqa: BLE001 - report, keep going
            rows.append({"id": ident, "global_id": "", "name": "", "old_class": "", "new_class": new_class,
                         "status": "error", "note": f"{type(e).__name__}: {e}"})
    f.write(str(dst))
    if changelog:
        with open(changelog, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["id"])
            w.writeheader()
            w.writerows(rows)
    log.info("reclass: %d ok, %d errors -> %s", sum(r["status"] == "ok" for r in rows),
             sum(r["status"] == "error" for r in rows), dst)
    return rows
