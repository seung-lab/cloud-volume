from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import python_jsonschema_objects as pjs
import orjson
import json5

dataset_provenance_schema: Dict[str, Any] = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "id": "DatasetProvenance",
  "title": "Dataset Provenance",
  "description": "Represents a dataset and its derived data layers.",
  "required": [
    "dataset_name", "dataset_description",
    "organism", "imaged_date", "imaged_by",
    "owners"
  ],
  "properties": {
    'dataset_name': { 'type': 'string' },
    'dataset_description': { 'type': 'string' },
    'organism': {
      'type': 'string',
      'description': 'Species, sex, strain identifier',
    },
    'imaged_date': { 'type': 'string' },
    'imaged_by': { 'type': 'string' },
    'references': { # e.g. dois, urls, titles
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 0,
      "uniqueItems": True, # e.g. email addresses
    },
    'owners': {
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": True, # e.g. email addresses
    }
  }
}

builder = pjs.ObjectBuilder(dataset_provenance_schema)
classes = builder.build_classes()
DatasetProvenance = classes.DatasetProvenance

layer_provenance_schema: Dict[str, Any] = {
  "$schema": "http://json-schema.org/draft-04/schema#",
  "id": "DataLayerProvenance",
  "title": "Data Layer Provenance",
  "description": "Represents a data layer within a dataset. e.g. image, segmentation, etc",
  "required": [
    'description', 'sources',
    'processing', 'owners'
  ],
  "properties": {
    'description': { 'type': 'string' },
    'sources': { # e.g. [ 'gs://neuroglancer/pinky40_v11/image'
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 0,
      "uniqueItems": True,
    },
    'processing': {
      "type": "array",
      "items": {
        "type": "object"
      },
      "minItems": 0,
    },
    # e.g. processing = [
    #    { 'method': 'inceptionnet', 'by': 'example@princeton.edu' },
    #    { 'method': 'downsample', 'by': 'example2@princeton.edu', 'description': 'demo of countless downsampling' }
    #    { 'method': 'meshing', 'by': 'example2@princeton.edu', 'description': '512x512x512 mip 3 simplification factor 30' }
    # ]
    'owners': {
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 0,
      "uniqueItems": True,
    },
  }
}

builder = pjs.ObjectBuilder(layer_provenance_schema)
classes = builder.build_classes()
DataLayerProvenanceValidation = classes.DataLayerProvenance

class DataLayerProvenance(dict):
  def __init__(self, *args: Any, **kwargs: Any) -> None:
    dict.__init__(self, *args, **kwargs)
    if 'description' not in self:
      self['description'] = ''
    if 'owners' not in self:
      self['owners'] = []
    if 'processing' not in self:
      self['processing'] = []
    if 'sources' not in self:
      self['sources'] = ''

  def validate(self) -> None:
    DataLayerProvenanceValidation(**self).validate()

  def serialize(self) -> str:
    return orjson.dumps(self).decode("utf8")

  def from_json(self, data: Union[str, bytes]) -> DataLayerProvenance:
    data = json5.loads(data)
    self.update(data)
    return self

  @classmethod
  def create(cls, mydict: Dict[str, Any]) -> Any:
    return DatasetProvenance(**mydict)

  @property
  def description(self) -> str:
    return self['description']
  @description.setter
  def description(self, val: str) -> None:
    self['description'] = val

  @property
  def owners(self) -> List[str]:
    return self['owners']
  @owners.setter
  def owners(self, val: List[str]) -> None:
    self['owners'] = val

  @property
  def processing(self) -> List[Dict[str, Any]]:
    return self['processing']
  @processing.setter
  def processing(self, val: List[Dict[str, Any]]) -> None:
    self['processing'] = val

  @property
  def sources(self) -> Union[str, List[str]]:
    return self['sources']
  @sources.setter
  def sources(self, val: Union[str, List[str]]) -> None:
    self['sources'] = val


__all__ = [
  DataLayerProvenance
]
