"""Frozen-data runtime helpers. Adapted from MIT-licensed sources; see LICENSE."""
from __future__ import annotations

from collections import Counter

from copy import deepcopy

import re

from .language import validate_public_wording
from ..artifacts.layout import json_digest as digest

SPATIAL = re.compile(r'\b(in|on|at|near|beside|behind|between|under|above|below|next|left|right|front|back|corner|wall|window|door|closest|tallest|shortest)\b', re.I)

ATTRIBUTE_KEYS = ('color', 'material', 'shape', 'pattern')


def apply_object_deltas(objects, deltas):
    """Replay captured object fields without confusing absent fields with null."""
    for delta in deltas:
        obj = objects[delta['object_id']]
        for key, change in delta['changes'].items():
            if (obj.get(key) != change['before'] or
                    ('before_present' in change and (key in obj) != change['before_present'])):
                raise ValueError('object delta has conflicting before state')
            if change.get('after_present', True):
                obj[key] = deepcopy(change['after'])
            else:
                obj.pop(key, None)


def label(category):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', str(category)).replace('_', ' ').lower()

def attributes(semantic):
    # Old caches have prose only. Do not infer attributes of a whole object
    # from the colour/material of a nearby landmark or of one of its parts.
    result = semantic.get('identity_attributes', {})
    return {k: v.strip().lower() for k, v in result.items()
            if k in ATTRIBUTE_KEYS and isinstance(v, str) and v.strip()}

def geometry(entity):
    """Translate metadata AABBs when a simulator stores pose separately."""
    result = deepcopy(entity.get('geometry', {}))
    pose = entity.get('pose', {}).get('position')
    matrix = result.get('world_matrix4x4')
    bounds = result.get('world_aabb')
    if pose and matrix and bounds:
        delta = [float(pose[i]) - float(matrix[3][i]) for i in range(3)]
        for side in ('min', 'max'):
            bounds[side] = [bounds[side][i] + delta[i] for i in range(3)]
        matrix[3][:3] = list(pose[:3])
    return result

def location_signature(entity):
    return {k: deepcopy(entity.get(k)) for k in ('room_id', 'support_object_id', 'held')} | {
        'geometry': geometry(entity)}

def build_runtime_cache(metadata, understanding, profile=None):
    profile = profile or {}
    semantic = {o['object_id']: o for o in understanding['objects']}
    objects = metadata['objects']
    counts = Counter(o.get('category') for o in objects)
    local = Counter((o.get('room_id'), o.get('category')) for o in objects)
    result = {'schema_version': 1, 'scene_id': metadata['scene']['scene_id'],
              'source_cache_key': understanding.get('cache_key'), 'objects': {}, 'areas': {}}
    for obj in objects:
        oid, category = obj['object_id'], obj.get('category')
        sem = semantic.get(oid, {})
        expressions = []
        for expression in sem.get('referring_expressions', []):
            text = expression.get('text', '')
            if not expression.get('unique'):
                continue
            try:
                validate_public_wording(text)
            except ValueError:
                continue
            anchors = [other['object_id'] for other in objects if other.get('category') != category
                       and re.search(r'\b' + re.escape(label(other.get('category'))) + r'\b', text, re.I)]
            anchors.extend(rel['target_object_id'] for rel in sem.get('relations', []) if rel.get('target_object_id'))
            expressions.append({'text': text, 'binding': expression.get('binding'),
                                'source': expression.get('source', 'visual_cache'),
                                'spatial': bool(SPATIAL.search(text)),
                                'dependency_object_ids': sorted(set(anchors))})
        result['objects'][oid] = {
            'category': category, 'category_name': label(category),
            'identity_attributes': attributes(sem), 'appearance': sem.get('appearance', ''),
            'initial': location_signature(obj), 'expressions': expressions,
            'unique_in_scene': counts[category] == 1,
            'unique_in_initial_area': local[(obj.get('room_id'), category)] == 1,
            'relations': deepcopy(sem.get('relations', []))}
    rooms = {r['room_id']: r for r in understanding['rooms']}
    for room in metadata['rooms']:
        rid = room['room_id']
        entry = profile.get('areas', {}).get(rid, {})
        name = entry.get('en', room.get('public_name', label(room.get('room_type', rid))))
        validate_public_wording(name)
        # Descriptive area names may mention objects in an adjoining room.
        # All explicitly named category anchors must retain their locations.
        anchors = [o['object_id'] for o in objects
                   if re.search(r'\b' + re.escape(label(o.get('category'))) + r'\b', name, re.I)]
        result['areas'][rid] = {'text': name, 'description': rooms.get(rid, {}).get('description', ''),
                                'anchor_object_ids': anchors,
                                'evidence': entry.get('evidence', 'scene room metadata')}
    result['cache_key'] = digest(result)
    return result

def same_location(current, initial):
    current = location_signature(current)
    # False/absent held both mean the object is not being held.
    current['held'] = bool(current['held'])
    initial = deepcopy(initial)
    initial['held'] = bool(initial.get('held'))
    return current == initial
