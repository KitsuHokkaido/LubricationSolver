from .engine import Geometry, LubricationSolver

def main():
    solver = LubricationSolver()
    solver.solve(geometry_type=Geometry.SQUEEZE)
