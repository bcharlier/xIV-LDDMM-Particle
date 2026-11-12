"""
Test
========================================
"""
import pyvista as pv

pl = pv.Plotter(off_screen=True)
pl.add_mesh(pv.Sphere(), color="orange")
pl.export_html("test_scene.html")
pl.show(auto_close=False)