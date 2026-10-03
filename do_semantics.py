from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import OWL, RDF, RDFS

DO_PREFIX = "http://purl.obolibrary.org/obo/DOID_"
OBOINOWL = "http://www.geneontology.org/formats/oboInOwl#"
IAO_DEF = "http://purl.obolibrary.org/obo/IAO_0000115"


BREAKHIS_CLASSES = [
    "adenosis",
    "ductal_carcinoma",
    "fibroadenoma",
    "lobular_carcinoma",
    "mucinous_carcinoma",
    "papillary_carcinoma",
    "phyllodes_tumor",
    "tubular_adenoma",
]


def _dedupe(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _local_id(uri: URIRef) -> str | None:
    text = str(uri)
    if text.startswith(DO_PREFIX):
        return "DOID:" + text[len(DO_PREFIX) :]
    return None


def _is_do_class(uri: URIRef) -> bool:
    return _local_id(uri) is not None


def _literal_text(value: Literal | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_doid_from_literal(value: Literal | None) -> str | None:
    text = _literal_text(value)
    if not text:
        return None
    match = re.search(r"DOID:\d+(?:\.\d+)?", text)
    return match.group(0) if match else None


@dataclass(frozen=True)
class Concept:
    doid: str
    uri: str
    label: str
    synonyms: list[str]
    definition: str | None
    parents: list[str]
    ancestors: list[str]


@dataclass(frozen=True)
class SemanticRecord:
    breakhis_label: str
    status: str
    doid: str | None
    semantic_text: str
    concept_label: str | None
    synonyms: list[str]
    definition: str | None
    parents: list[str]
    ancestors: list[str]
    provenance: dict[str, Any]


def _build_axiom_annotations(g: Graph) -> dict[tuple[str, str], list[Literal]]:
    out: dict[tuple[str, str], list[Literal]] = {}
    annotated_source = OWL.annotatedSource
    annotated_property = OWL.annotatedProperty
    annotated_target = OWL.annotatedTarget

    for axiom in g.subjects(annotated_source, None):
        source = g.value(axiom, annotated_source)
        prop = g.value(axiom, annotated_property)
        target = g.value(axiom, annotated_target)
        if isinstance(source, URIRef) and isinstance(prop, URIRef) and isinstance(target, Literal):
            out.setdefault((str(source), str(prop)), []).append(target)
    return out


def load_do_ontology(ontology_path: Path) -> tuple[Graph, dict[str, Concept]]:
    g = Graph()
    g.parse(str(ontology_path), format="xml")

    axioms = _build_axiom_annotations(g)
    concepts: dict[str, Concept] = {}

    for uri in g.subjects(RDF.type, OWL.Class) if False else []:
        pass
    from rdflib.namespace import RDF

    for uri in g.subjects(RDF.type, OWL.Class):
        if not isinstance(uri, URIRef) or not _is_do_class(uri):
            continue

        doid = _local_id(uri)
        assert doid is not None

        label = _literal_text(g.value(uri, RDFS.label)) or doid

        synonyms: list[str] = []
        syn_pred = URIRef(OBOINOWL + "hasExactSynonym")
        synonyms.extend(
            _literal_text(obj)
            for obj in g.objects(uri, syn_pred)
            if isinstance(obj, Literal)
        )
        for pred_name in (
            "hasRelatedSynonym",
            "hasBroadSynonym",
            "hasNarrowSynonym",
        ):
            pred = URIRef(OBOINOWL + pred_name)
            synonyms.extend(
                _literal_text(obj)
                for obj in g.objects(uri, pred)
                if isinstance(obj, Literal)
            )
        definition_values: list[str] = []
        definition_pred = URIRef(IAO_DEF)

        for obj in g.objects(uri, definition_pred):
            if isinstance(obj, Literal):
                text = _literal_text(obj)
                if text:
                    definition_values.append(text)

        for value in axioms.get((str(uri), IAO_DEF), []):
            text = _literal_text(value)
            if text:
                definition_values.append(text)

        definition = _dedupe(definition_values)[0] if _dedupe(definition_values) else None
        parent_uris = [
            obj
            for obj in g.objects(uri, RDFS.subClassOf)
            if isinstance(obj, URIRef) and _is_do_class(obj)
        ]

        parents = [
            _local_id(parent) for parent in parent_uris
            if _local_id(parent) is not None
        ]
        ancestors: list[str] = []
        frontier = [(parent, 1) for parent in parent_uris]
        visited: set[URIRef] = {uri}

        while frontier:
            current, depth = frontier.pop(0)
            if current in visited or depth > 2:
                continue
            visited.add(current)
            current_doid = _local_id(current)
            if current_doid:
                ancestors.append(current_doid)
            for parent in g.objects(current, RDFS.subClassOf):
                if isinstance(parent, URIRef) and _is_do_class(parent):
                    frontier.append((parent, depth + 1))

        concepts[doid] = Concept(
            doid=doid,
            uri=str(uri),
            label=label,
            synonyms=_dedupe(synonyms),
            definition=definition,
            parents=_dedupe(parents),
            ancestors=_dedupe(ancestors),
        )

    return g, concepts


def load_breakhis_mapping(mapping_path: Path) -> dict[str, dict[str, Any]]:
    data = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Mapping YAML must contain a top-level mapping.")

    if "mappings" in data and isinstance(data["mappings"], dict):
        data = data["mappings"]
    elif "classes" in data and isinstance(data["classes"], dict):
        data = data["classes"]

    out: dict[str, dict[str, Any]] = {}
    for label, spec in data.items():
        if isinstance(spec, str):
            out[str(label)] = {"doid": spec, "status": "exact"}
        elif isinstance(spec, dict):
            out[str(label)] = dict(spec)
        elif spec is None:
            out[str(label)] = {"doid": None, "status": "unavailable"}
        else:
            raise ValueError(f"Unsupported mapping for {label!r}: {spec!r}")
    return out


def _semantic_text(
    label: str,
    concept: Concept | None,
    status: str,
) -> str:
    if concept is None:
        return f"breast disease class: {label.replace('_', ' ')}"

    parts = [f"disease: {concept.label}."]

    if concept.synonyms:
        parts.append("synonyms: " + "; ".join(concept.synonyms) + ".")

    if concept.definition:
        parts.append("definition: " + concept.definition)

    if concept.ancestors:
        parts.append(
            "ontology ancestors: "
            + " > ".join(
                _concept_display(a, concept_map=None)
                for a in concept.ancestors
            )
            + "."
        )

    if status != "exact":
        parts.append(f"mapping status: {status}.")

    return " ".join(parts)


def _concept_display(doid: str, concept_map: dict[str, Concept] | None) -> str:
    if concept_map and doid in concept_map:
        return concept_map[doid].label
    return doid


def build_semantic_records(
    ontology_path: Path,
    mapping_path: Path,
) -> dict[str, SemanticRecord]:
    _, concepts = load_do_ontology(ontology_path)
    mapping = load_breakhis_mapping(mapping_path)

    records: dict[str, SemanticRecord] = {}

    for label in BREAKHIS_CLASSES:
        spec = mapping.get(label, {"doid": None, "status": "unavailable"})
        status = str(spec.get("status", "exact"))
        doid = spec.get("doid")

        if doid is not None:
            doid = str(doid)
            concept = concepts.get(doid)
            if concept is None:
                raise ValueError(
                    f"Mapping for {label!r} points to {doid}, "
                    "but that DOID is absent from the supplied ontology."
                )
        else:
            concept = None
            status = "unavailable"
        if concept is not None:
            ancestor_labels = [
                concepts[a].label if a in concepts else a
                for a in concept.ancestors
            ]
            parts = [f"disease: {concept.label}."]
            if concept.synonyms:
                parts.append("synonyms: " + "; ".join(concept.synonyms) + ".")
            if concept.definition:
                parts.append("definition: " + concept.definition)
            if ancestor_labels:
                parts.append("ontology ancestors: " + " > ".join(ancestor_labels) + ".")
            if status != "exact":
                parts.append(f"mapping status: {status}.")
            semantic_text = " ".join(parts)
        else:
            semantic_text = (
                f"breast disease class: {label.replace('_', ' ')}. "
                "No direct DOID mapping is available in the supplied ontology."
            )

        records[label] = SemanticRecord(
            breakhis_label=label,
            status=status,
            doid=doid,
            semantic_text=semantic_text,
            concept_label=concept.label if concept else None,
            synonyms=concept.synonyms if concept else [],
            definition=concept.definition if concept else None,
            parents=concept.parents if concept else [],
            ancestors=concept.ancestors if concept else [],
            provenance={
                    "ontology_file": Path(ontology_path).name,
                "mapping_file": Path(mapping_path).name,
                "mapping_verified": bool(concept) if doid else False,
            },
        )

    return records


def save_semantic_records(
    records: dict[str, SemanticRecord],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ontology_class_count": None,
        "records": {
            key: asdict(value)
            for key, value in records.items()
        },
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
