"""Assemble the probe suite.

Two invariants this module has to hold, both load-bearing:

Content probes only go to restricted entities, because only they have a protected
field that could leak. Behavioural probes go to every entity with the identical
template set, because that symmetry is what makes the detectability comparison valid.
If restricted and control entities got different questions, a classifier separating
them would be reading the question, not the answer.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..types import CONTENT_FAMILIES, Corpus, Entity, Probe, ProbeFamily, make_probe_id
from .templates import BEHAVIOURAL, TEMPLATES_BY_FAMILY, ProbeTemplate

__all__ = ["build_probe_suite", "probes_for_entity"]


def _content_probe(
    template: ProbeTemplate,
    entity: Entity,
    corpus: Corpus,
    template_pack_version: str,
) -> Probe | None:
    """Build one content probe, or None when the deal lacks the targeted field."""
    deal = corpus.deal_for_entity(entity.entity_id)
    if deal is None or template.target_field is None:
        return None

    field = deal.field_by_name(template.target_field)
    if field is None:
        return None

    slots = {"name": entity.display_name}
    return Probe(
        probe_id=make_probe_id(
            template.family, template.template_id, template_pack_version, entity.entity_id, slots
        ),
        family=template.family,
        entity_id=entity.entity_id,
        template_id=template.template_id,
        prompt=template.render(entity.display_name),
        target_field_ids=(field.field_id,),
        tools_exposed=template.tools,
        memory_primed=template.memory_primed,
    )


def _behavioural_probe(
    template: ProbeTemplate, entity: Entity, template_pack_version: str
) -> Probe:
    slots = {"name": entity.display_name}
    return Probe(
        probe_id=make_probe_id(
            template.family, template.template_id, template_pack_version, entity.entity_id, slots
        ),
        family=ProbeFamily.BEHAVIOURAL,
        entity_id=entity.entity_id,
        template_id=template.template_id,
        prompt=template.render(entity.display_name),
        target_field_ids=(),
        tools_exposed=(),
        memory_primed=False,
    )


def build_probe_suite(
    corpus: Corpus,
    template_pack_version: str = "tp1",
    n_behavioural: int = 8,
    n_content_per_family: int = 3,
) -> tuple[Probe, ...]:
    """Full probe suite for a corpus, in a stable order."""
    probes: list[Probe] = []

    for entity in sorted(corpus.restricted, key=lambda e: e.entity_id):
        for family in CONTENT_FAMILIES:
            for template in TEMPLATES_BY_FAMILY[family][:n_content_per_family]:
                probe = _content_probe(template, entity, corpus, template_pack_version)
                if probe is not None:
                    probes.append(probe)

    # identical behavioural set for every entity, restricted and control alike
    for entity in sorted(corpus.entities, key=lambda e: e.entity_id):
        for template in BEHAVIOURAL[:n_behavioural]:
            probes.append(_behavioural_probe(template, entity, template_pack_version))

    probes.sort(key=lambda p: (p.entity_id, p.family.value, p.template_id))
    return tuple(probes)


def probes_for_entity(probes: Sequence[Probe], entity_id: str) -> tuple[Probe, ...]:
    return tuple(p for p in probes if p.entity_id == entity_id)
