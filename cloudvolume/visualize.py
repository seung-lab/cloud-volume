from typing import Sequence, Union

from collections import defaultdict

import numpy as np

import microviewer

from .lib import toiter
from .mesh import Mesh
from .skeleton import Skeleton
from .volumecutout import VolumeCutout

def is_mesh(obj):
  """zmesh meshes are fine, but not covered by cloudvolume.Mesh"""
  return isinstance(obj, Mesh) or (hasattr(obj, "vertices") and hasattr(obj, "faces"))

def view(
  objects:Sequence[Union[VolumeCutout, Mesh, Skeleton]],
):
  """
  Produce a 3D co-visualization of meshes and skeletons together.
  """
  objects = toiter(objects)

  pairs = defaultdict(list)

  for obj in objects:
    if isinstance(obj, VolumeCutout):
      microviewer.view(obj, seg=obj.is_segmentation)
      continue
    elif isinstance(obj, np.ndarray):
      microviewer.view(obj)
      continue

    pairs[obj.id].append(obj)

  actors = []
  for obj in pairs[None]:
    if is_mesh(obj):
      actors.append(
        create_vtk_mesh(obj, opacity=1.0)
      )
    elif isinstance(obj, Skeleton):
      actors.extend(
        create_vtk_skeleton(obj)
      )

  pairs.pop(None)

  segids = []
  for label, objs in pairs.items():
    segids.append(label)
    mesh_opacity = 1.0
    for obj in objs:
      if isinstance(obj, Skeleton):
        mesh_opacity = 0.2
        break

    for obj in objs:
      if is_mesh(obj):
        actors.append(
          create_vtk_mesh(obj, opacity=mesh_opacity)
        )
      elif isinstance(obj, Skeleton):
        actors.extend(
          create_vtk_skeleton(obj)
        )

  segids.sort()

  display_actors(segids, actors)

def display_actors(segids, actors):
  if len(actors) == 0:
    return

  try:
    import vtk
  except ImportError:
    print("The visualize viewer requires the OpenGL based vtk. Try: pip install vtk --upgrade")
    return

  renderer = vtk.vtkRenderer()
  renderer.SetBackground(0.1, 0.1, 0.1)

  render_window = vtk.vtkRenderWindow()
  render_window.AddRenderer(renderer)
  render_window.SetSize(2000, 1500)

  interactor = vtk.vtkRenderWindowInteractor()
  style = vtk.vtkInteractorStyleTrackballCamera()
  interactor.SetInteractorStyle(style)

  interactor.SetRenderWindow(render_window)

  for actor in actors:
    renderer.AddActor(actor)

  window_name = f"Mesh & Skeleton Viewer"
  if segids:
    segids = [ str(x) for x in segids ]
    window_name += f" ({','.join(segids)})"
  
  render_window.SetWindowName(window_name)
  
  render_window.Render()
  interactor.Start()

def create_vtk_skeleton(skel):
  import vtk.util.numpy_support

  colors = [
      [.8,.8,.8],
      [0,0.6,0],
      [1,0,0],
      [0,0,1],
      [1,0,1],
      [0,1,1],
      [1,1,0],
    ]

  actors = []

  skels = skel.components()

  for i, skel in enumerate(skels):
    points = vtk.vtkPoints()
    points.SetData(vtk.util.numpy_support.numpy_to_vtk(skel.vertices))

    lines = vtk.vtkCellArray()

    for edge in skel.edges:
      line = vtk.vtkLine()
      line.GetPointIds().SetId(0, edge[0])
      line.GetPointIds().SetId(1, edge[1])
      lines.InsertNextCell(line)
  
    polyline = vtk.vtkPolyData()
    polyline.SetPoints(points)
    polyline.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyline)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(2)

    color = colors[i % len(colors)]
    actor.GetProperty().SetColor(*color)
    actors.append(actor)

  return actors

def create_vtk_mesh(mesh, opacity=1.0):
  import vtk
  from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

  vertices = mesh.vertices
  faces = mesh.faces

  vtk_points = vtk.vtkPoints()
  vtk_points.SetData(numpy_to_vtk(vertices))
  
  polydata = vtk.vtkPolyData()
  polydata.SetPoints(vtk_points)
  
  vtk_faces = vtk.vtkCellArray()
  vtk_faces.SetCells(
    faces.shape[0], 
    numpy_to_vtkIdTypeArray(
      np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
    )
  )

  polydata.SetPolys(vtk_faces)
  
  mapper = vtk.vtkPolyDataMapper()
  mapper.SetInputData(polydata)

  actor = vtk.vtkActor()
  actor.SetMapper(mapper)

  actor.GetProperty().SetOpacity(opacity)

  return actor





