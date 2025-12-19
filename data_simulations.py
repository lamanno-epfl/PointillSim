import numpy as np
import scipy as sp
import pandas as pd
import skimage as ski
import matplotlib
import matplotlib.pyplot as plt
import tqdm
from shapely.geometry import Point, Polygon, LinearRing, LineString
from shapely.affinity import scale
import os
import glob
from collections import defaultdict
from collections import OrderedDict
import time
from sklearn.neighbors import KDTree
import copy


__all__ = [
    "generate_uniform_points_in_circle",
    "generate_points_asin_cell",
    "one_hot_encode_array",
    "lognorm_params_to_mean_std",
    "intuitive_rand_lognormal",
    "unfold_int_matrix",
    "generate_dataset",
    "load_data",
    "LinearNDInterpolatorExt",
    # Histological elements
    "HistologicalElement",
    "FrameWideElement",
    "FrameWideUpdater",
    # Main Objects
    "FOVDistribution",
    "FOV",
    "CellTypesProperties",
    "TissueCellTypes",
    "HybISS_Setup",
    # Rules
    "CellTypeRuleBase",
    "RandomCellTypeRule",
    "SingleTypeRule",
    "ProbabilityNodeFieldRule",
    "DeterministicNeighborAssignment",
    "DummyRule",
    "VacuolatedStructure",
    "MixOfNCellTypesRule",
    "AffineNonNegTransfer",
    "IdentityTransfer",
]


def generate_uniform_points_in_circle(center, scale, num_points):
    """Generate points uniformly distributed in a circle"""

    points = np.random.normal(
        0, 1, (num_points, center.shape[-1])
    )  # Draw points from a standard normal distribution
    samples = (
        0.5
        * np.sqrt(np.random.uniform(0, scale**2, (num_points, 1)))
        * points
        / np.linalg.norm(points, axis=1, keepdims=True)
    )
    points = center + samples
    return points


def generate_points_asin_cell(
    center,
    scale,
    num_points,
    major_axis,
    minor_axis,
    rotation_angle,
    center_balance=0.4,
    smeer=0.4,
):
    """Generate points within an ellipse with some noise"""
    # Broadcasting in the real use case is not possible unless I add a 3rd dimension
    points = np.random.normal(0, 1, (num_points, center.shape[-1]))
    renorm = np.linalg.norm(points, axis=1, keepdims=True)
    samples = (
        (0.5)
        * np.random.uniform(
            0, (scale * (1 + smeer / 2.0)) ** (2.0 / center_balance), (num_points, 1)
        )
        ** (center_balance * 0.5)
        * points
        / (renorm + smeer)
    )
    samples[:, 0] = samples[:, 0] * major_axis
    samples[:, 1] = samples[:, 1] * minor_axis
    # Rotate the points
    points = np.zeros_like(samples)
    points[:, 0] = samples[:, 0] * np.cos(rotation_angle) - samples[:, 1] * np.sin(
        rotation_angle
    )
    points[:, 1] = samples[:, 0] * np.sin(rotation_angle) + samples[:, 1] * np.cos(
        rotation_angle
    )
    return center + samples


def one_hot_encode_array(class_ix, num_classes):
    """One hot encode an array of class indices"""
    one_hot_matrix = np.zeros((len(class_ix), num_classes))
    one_hot_matrix[np.arange(len(class_ix)), class_ix] = 1
    return one_hot_matrix


def lognorm_params_to_mean_std(mu, sigma):
    """Convert lognormal parameters to mean and standard deviation of the normal distribution"""
    return np.log(mu) - 0.5 * np.log(1 + (sigma / mu) ** 2), np.sqrt(
        np.log(1 + (sigma / mu) ** 2)
    )


def intuitive_rand_lognormal(mu, sigma, size):
    """Generate a random lognormal distribution with a given mean and standard deviation"""
    return np.random.lognormal(*lognorm_params_to_mean_std(mu, sigma), size)


def unfold_int_matrix(a):
    """Unfold a matrix of integers into a list of indices"""
    unfolded_0 = []
    unfolded_1 = []
    for i in range(a.shape[0]):
        unfolded_per_row1 = []
        unfolded_per_row0 = []
        for j in range(a.shape[1]):
            for k in range(a[i, j]):
                unfolded_per_row1.append(j)
                unfolded_per_row0.append(i)
        unfolded_1.append(unfolded_per_row1)
        unfolded_0.append(unfolded_per_row0)
    return unfolded_0, unfolded_1


class LinearNDInterpolatorExt(sp.interpolate.LinearNDInterpolator):
    """LinearNDInterpolator with a fallback to RBFInterpolator for points outside the convex hull"""

    def __init__(self, points, values):
        super().__init__(points, values)
        self.funcinterp = sp.interpolate.LinearNDInterpolator(points, values)
        self.funcnearest = sp.interpolate.RBFInterpolator(points, values)

    def __call__(self, *args):
        t = self.funcinterp(*args)
        return np.where(
            np.any(np.isnan(t), axis=1)[:, None], self.funcnearest(*args), t
        )


class CellTypeRuleBase:
    """Base class for rules that assign cell types to cells in a histological element"""

    def __init__(self, n_cell_types=3):
        self.n_cell_types = n_cell_types
        self.is_adapted = False

    def apply(self, cell_centroids, current_probs=None, **kwargs):
        """Apply the rule to the cell centroids and return the probabilities of each cell type for each cell"""
        raise NotImplementedError()

    def adapt_rule_to_element(self, element):
        """Adapt the rule to the element by setting the reference points or other parameters of the rule"""
        self.is_adapted = True
        return self


class RandomCellTypeRule(CellTypeRuleBase):
    """Rule that assigns cell types to cells randomly"""

    def __init__(self, n_cell_types=3):
        super().__init__(n_cell_types)

    def apply(self, cell_centroids, current_probs=None):
        """Assign cell types probability to cells randomly with a Dirichlet distribution"""
        if current_probs is None:
            probs = np.random.dirichlet(np.ones(self.n_cell_types), len(cell_centroids))
        else:
            # this is stupid, but it's just to make an example of how different rules could compose
            probs = current_probs + np.random.normal(0, 0.1, current_probs.shape)
            # renormalize
            probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs


class MixOfNCellTypesRule(CellTypeRuleBase):
    def __init__(self, n_cell_types=3, N=None, list_N=None, proportions=None):
        super().__init__(n_cell_types)

        if list_N is None and N is not None:
            self.N = N
            self.list_N = np.random.choice(np.arange(N), replace=False)
        elif list_N is not None and (N is None or N == len(list_N)):
            self.list_N = np.array(list_N)
        else:
            raise ValueError("N a d list_N must needs to be consistent")
        # randomly draw proportions from
        if proportions is not None:
            self.proportions = np.array(proportions)
        else:
            self.proportions = np.random.dirichlet(np.ones(len(self.list_N)), 1)[0]
        self.probs = np.zeros(self.n_cell_types)
        self.probs[self.list_N] = self.proportions

    def apply(self, cell_centroids, current_probs=None):
        """Assign cell types probability to cells randomly with a Dirichlet distribution"""
        probs = self.probs * np.ones(cell_centroids.shape[0])[:, None]
        if current_probs is None:
            return probs
        else:
            # this is stupid, but it's just to make an example of how different rules could compose
            probs = current_probs + probs
            # renormalize
            probs = probs / np.sum(probs, axis=1, keepdims=True)
            return probs


class ProbabilityNodeFieldRule(CellTypeRuleBase):
    """Rule that assigns cell types to cells based on a probability field defined by reference points"""

    def __init__(
        self,
        n_cell_types=3,
        n_ref_points=4,
        alpha=0.05,
        ref_probs=None,
        element=None,
        reference_points=None,
    ):
        """It is a rule that assigns cell types to cells based on a probability field defined by reference points

        Parameters
        ----------
        n_cell_types : int
            The number of cell types
        n_ref_points : int
            The number of reference points
        alpha : float
            The alpha parameter of the Dirichlet distribution
            higher values make the probabilities more uniform
            lower values make the probabilities more peaked
        ref_probs : np.ndarray
            The probabilities of each cell type at the reference points
        element : HistologicalElement
            The element to adapt the rule to
        reference_points : np.ndarray
            The reference points to use for the rule
        """
        super().__init__(n_cell_types)
        self.n_ref_points = n_ref_points
        self.alpha = alpha
        if ref_probs is None:
            self.ref_probs = np.random.dirichlet(
                self.alpha * np.ones(self.n_cell_types), self.n_ref_points
            )
        else:
            self.ref_probs = ref_probs
        if element is not None:
            self.adapt_rule_to_element(element)
        if reference_points is not None:
            self.reference_points = reference_points
            self.is_adapted = True  # adaptation for this rule is just the position of the reference points
        else:
            self.reference_points = None

    def adapt_rule_to_element(self, element):
        """Adapt the rule to the element by setting the reference points outside the element"""
        if self.reference_points is not None:
            self.is_adapted = True
            return self  # <-- ADD THIS LINE
        
        # This code will now only run if reference_points were NOT provided
        self.is_adapted = True
        vertices = generate_uniform_points_in_circle(
            element.center, element.scale, self.n_ref_points
        )
        # these vertices could be messed up
        if self.n_ref_points <= 2:
            self.reference_points = vertices
        else:
            vertices = np.row_stack([vertices, vertices[0:1, :]])
            cvxh = np.array(Polygon(vertices).convex_hull.exterior.coords)[:-1]
            if cvxh.shape[0] >= self.n_ref_points:
                polygon_vertices = cvxh[: self.n_ref_points]
            elif self.n_ref_points - cvxh.shape[0] == 1:
                polygon_vertices = np.row_stack(
                    [cvxh, (cvxh[-1:, :] + cvxh[:1, :]) / 2.0]
                )
            elif self.n_ref_points - cvxh.shape[0] == 2:
                polygon_vertices = np.row_stack(
                    [cvxh, (cvxh[-1:, :] + cvxh[:1, :]) / 2.0]
                )
                polygon_vertices = np.row_stack(
                    [
                        polygon_vertices,
                        (polygon_vertices[-1:, :] + polygon_vertices[:1, :]) / 2.0,
                    ]
                )
            else:
                raise ValueError("Weird corner case")
            self.reference_points = polygon_vertices
        return self

    def apply(self, cell_centroids, current_probs=None, mix=0.9):
        """Assign cell types probability to cells based on a probability field defined by reference points"""
        if self.is_adapted is False:
            raise ValueError("Rule not adapted to element")
        else:
            # Create the interpolator
            # interpolator = sp.interpolate.RBFInterpolator(
            #     self.reference_points, self.ref_probs, kernel="linear"
            # )
            interpolator = LinearNDInterpolatorExt(self.reference_points, self.ref_probs)
            # LinearNDInterpolatorExt(self.reference_points, self.ref_probs)
            # Interpolate the probabilities for each cell
            interpolated_probs = interpolator(cell_centroids)
            if np.isnan(interpolated_probs).any():
                plt.plot(self.reference_points[:, 0], self.reference_points[:, 1], "-o")
                raise ValueError("Interpolated probabilities contain NaNs")
            if not current_probs is None:
                # stupidly combine them
                interpolated_probs = (
                    mix * interpolated_probs + (1 - mix) * current_probs
                )
            interpolated_probs = np.maximum(interpolated_probs, 1e-12)

            # Normalize the probabilities (as operation above are not guaranteed in the simplex)
            cell_probs = interpolated_probs / (
                interpolated_probs.sum(axis=1)[:, np.newaxis] + 1e-8
            )

            return cell_probs


class SingleTypeRule(CellTypeRuleBase):
    """Rule that assigns a single cell type to all cells"""

    def __init__(self, n_cell_types, cell_type_ix=None):
        super().__init__(n_cell_types=n_cell_types)
        if cell_type_ix is None:
            self.cell_type_ix = np.random.randint(self.n_cell_types)
        else:
            self.cell_type_ix = cell_type_ix

    def apply(self, cell_centroids, current_probs=None, mix=0.9):
        """Assign a single cell type for each cell centroid"""
        new_probs = np.zeros((len(cell_centroids), self.n_cell_types), dtype=float)
        new_probs[:, self.cell_type_ix] = 0.9999999
        if current_probs is None:
            cell_probs = new_probs
        else:
            cell_probs = mix * new_probs + (1 - mix) * current_probs
        cell_probs = np.maximum(cell_probs, 1e-12)

        # Normalize the probabilities (as operation above are not guaranteed in the simplex)
        cell_probs = cell_probs / (cell_probs.sum(axis=1)[:, np.newaxis] + 1e-8)

        return cell_probs


class HistologicalElement:
    def __init__(
        self,
        frame_size=5000,
        n_vertices=(10, 15),
        scale=200,
        fixed_center=None,
        tipical_cell_spacing=8,
        rules=None,
    ):
        self.n_vertices = n_vertices
        self.frame_size = frame_size
        self.scale = scale
        self.fixed_center = fixed_center
        self.tipical_cell_spacing = tipical_cell_spacing
        self.polygon = None
        self.cell_centroids = None

        if rules is None:
            raise ValueError("No rules provided")
        self.rules = rules
        if not isinstance(self.rules, list):
            self.rules = [self.rules]
        self.original_rules = self.rules
        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    def generate(self, **kwargs):
        # reinitialize the element
        other = copy.deepcopy(self)
        other.polygon = None
        other.cell_centroids = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        # start the generation
        other.polygon = other.generate_bounding_polygon()
        other.cell_centroids = other.generate_cell_centroids()
        other.rules = [i.adapt_rule_to_element(other) for i in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)
        return other

    def generate_bounding_polygon(self):
        """Generate a bounding polygon for the histological element"""
        num_points = (
            np.random.randint(self.n_vertices[0], self.n_vertices[1])
            if isinstance(self.n_vertices, tuple)
            else self.n_vertices
        )
        if self.fixed_center is None or self.fixed_center is False:
            self.center = np.random.uniform(
                self.frame_size * 0.1, self.frame_size * 0.9, (1, 2)
            )
        else:
            self.center = self.fixed_center
        # points uniformelly distributed withing a circle of center and radius=scale
        points = generate_uniform_points_in_circle(self.center, self.scale, num_points)
        polygon = Polygon(points).convex_hull  # Create a convex hull
        return polygon

    @property
    def bounding_box(self):
        return self.polygon.bounds

    @property
    def class_instance(self):
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self):
        if self._class_instance_one_hot is None:
            self._class_instance_one_hot = self.rng.multinomial(
                n=1, pvals=self.cell_probabilities
            )
        return self._class_instance_one_hot

    @property
    def ML_class(self):
        return np.argmax(self.cell_probabilities, axis=1)

    def is_inside(self, points):
        # return a boolean array indicating if the points are inside the polygon
        return np.array(
            [self.polygon.contains(Point(point[0], point[1])) for point in points],
            dtype=bool,
        )

    def is_outside(self, points):
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix):
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]

    def generate_cell_centroids(self):
        """Generate a set of points within the polygon
        - the function generates a grid within the polygon
        - the function removes points that are outside the polygon
        - some jitter is added to the points to make them look more natural

        Note the grid has alternating rows offset by half a cell size,
        this kind of arrangement can be though as a hexagonal grid"""

        # Generate a grid of points
        x = np.arange(
            self.bounding_box[0], self.bounding_box[2], self.tipical_cell_spacing
        )
        y = np.arange(
            self.bounding_box[1],
            self.bounding_box[3],
            self.tipical_cell_spacing * np.sin(np.pi / 3),
        )
        X, Y = np.meshgrid(x, y)
        X[::2] += self.tipical_cell_spacing / 2.0
        points = np.stack((X.flatten(), Y.flatten()), axis=1)

        # Remove points outside the polygon
        points = points[self.is_inside(points)]

        # Add jitter
        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)
        return points

    def apply_rules(self, rules):
        probs = None
        for rule in rules:  # <-- FIX: Must iterate over the 'rules' argument, not 'self.rules'
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        # If no rules were applied (e.g., rules=[]), probs is still None.
        # We must return an array, not None, for slicing to work.
        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = 0  # Default
            if rules:  # <-- FIX: Must check the 'rules' argument, not 'self.rules'
                # Attempt to get n_cell_types from the first rule if it exists
                # We add a fallback in case rules[0] doesn't have n_cell_types
                try:
                    n_types = rules[0].n_cell_types
                except (IndexError, AttributeError):
                    n_types = 0 # Default to 0 if no rules or rule has no n_cell_types
            
            # Return an array with the correct dimensions (e.g., shape (N, 0))
            probs = np.empty((n_cells, n_types))

        return probs


class FrameWideElement(HistologicalElement):
    """A histological element that spans the whole frame"""

    def __init__(
        self,
        frame_size=5000,
        n_vertices=4,
        scale=5000,
        center=None,
        tipical_cell_spacing=8,
        rules=None,
    ):
        super().__init__(
            frame_size, n_vertices, scale, center, tipical_cell_spacing, rules
        )

    def generate_bounding_polygon(self):
        """
        Generates a bounding polygon for the frame.

        Returns:
            Polygon: The bounding polygon defined by the frame size.
        """
        self.center = np.array([self.frame_size / 2, self.frame_size / 2])
        return Polygon(
            [
                (0, 0),
                (self.frame_size, 0),
                (self.frame_size, self.frame_size),
                (0, self.frame_size),
                (0, 0),
            ]
        )


class VacuolatedStructure(HistologicalElement):
    def __init__(
        self,
        frame_size=5000,
        n_vertices=(10, 15),
        scale=200,
        fixed_center=None,
        tipical_cell_spacing=8,
        hole_scale_factor=0.75,
        rules=None,
    ):
        self.n_vertices = n_vertices
        self.frame_size = frame_size
        self.scale = scale
        self.fixed_center = fixed_center
        self.tipical_cell_spacing = tipical_cell_spacing
        self.hole_scale_factor = hole_scale_factor
        self.polygon = None
        self.hole = None
        self.cell_centroids = None

        if rules is None:
            raise ValueError("No rules provided")
        self.rules = rules
        if not isinstance(self.rules, list):
            self.rules = [self.rules]
        self.original_rules = self.rules
        self.rng = np.random.default_rng(seed=int(time.time() * 1e6))
        self._class_instance_one_hot = None

    def generate(self, **kwargs):
        # reinitialize the element
        other = copy.deepcopy(self)
        other.polygon = None
        other.hole = None
        other.cell_centroids = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        # start the generation
        other.polygon, other.hole = other.generate_polygon_andhole()
        other.cell_centroids = other.generate_cell_centroids()
        other.rules = [i.adapt_rule_to_element(other) for i in other.rules]
        other.cell_probabilities = other.apply_rules(other.rules)
        return other

    def generate_polygon_andhole(self):
        """Generate a bounding polygon for the histological element"""
        num_points = (
            np.random.randint(self.n_vertices[0], self.n_vertices[1])
            if isinstance(self.n_vertices, tuple)
            else self.n_vertices
        )
        if self.fixed_center is None or self.fixed_center is False:
            self.center = np.random.uniform(
                self.frame_size * 0.1, self.frame_size * 0.9, (1, 2)
            )
        else:
            self.center = self.fixed_center
        # points uniformelly distributed withing a circle of center and radius=scale
        points = generate_uniform_points_in_circle(self.center, self.scale, num_points)
        polygon = Polygon(points).convex_hull  # Create a convex hull
        hole = scale(
            polygon,
            xfact=self.hole_scale_factor,
            yfact=self.hole_scale_factor,
            origin=polygon.centroid,
        )
        return polygon, hole

    @property
    def bounding_box(self):
        return self.polygon.bounds

    @property
    def class_instance(self):
        return np.argmax(self.class_instance_one_hot, axis=1)

    @property
    def class_instance_one_hot(self):
        if self._class_instance_one_hot is None:
            self._class_instance_one_hot = self.rng.multinomial(
                n=1, pvals=self.cell_probabilities
            )
        return self._class_instance_one_hot

    @property
    def ML_class(self):
        return np.argmax(self.cell_probabilities, axis=1)

    def is_inside(self, points, consider_hole=False):
        if consider_hole:
            list_bool = []
            for point in points:
                px = Point(point[0], point[1])
                list_bool.append(
                    self.polygon.contains(px) and not self.hole.contains(px)
                )
            return np.array(list_bool, dtype=bool)
        else:
            return np.array(
                [self.polygon.contains(Point(point[0], point[1])) for point in points],
                dtype=bool,
            )

    def is_outside(self, points):
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix):
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]

    def generate_cell_centroids(self):
        """Generate a set of points within the polygon
        - the function generates a grid within the polygon
        - the function removes points that are outside the polygon
        - some jitter is added to the points to make them look more natural

        Note the grid has alternating rows offset by half a cell size,
        this kind of arrangement can be though as a hexagonal grid"""

        # Generate a grid of points
        x = np.arange(
            self.bounding_box[0], self.bounding_box[2], self.tipical_cell_spacing
        )
        y = np.arange(
            self.bounding_box[1],
            self.bounding_box[3],
            self.tipical_cell_spacing * np.sin(np.pi / 3),
        )
        X, Y = np.meshgrid(x, y)
        X[::2] += self.tipical_cell_spacing / 2.0
        points = np.stack((X.flatten(), Y.flatten()), axis=1)

        # Remove points outside the polygon
        points = points[self.is_inside(points, consider_hole=True)]

        # Add jitter
        points += np.random.normal(0, self.tipical_cell_spacing / 5.0, points.shape)
        return points

    def apply_rules(self, rules):
        probs = None
        for rule in rules:  # <-- FIX: Must iterate over the 'rules' argument, not 'self.rules'
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        # If no rules were applied (e.g., rules=[]), probs is still None.
        # We must return an array, not None, for slicing to work.
        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = 0  # Default
            if rules:  # <-- FIX: Must check the 'rules' argument, not 'self.rules'
                # Attempt to get n_cell_types from the first rule if it exists
                # We add a fallback in case rules[0] doesn't have n_cell_types
                try:
                    n_types = rules[0].n_cell_types
                except (IndexError, AttributeError):
                    n_types = 0 # Default to 0 if no rules or rule has no n_cell_types
            
            # Return an array with the correct dimensions (e.g., shape (N, 0))
            probs = np.empty((n_cells, n_types))

        return probs


class FOVDistribution:
    """A distribution of FOVs that can be sampled from"""

    def __init__(
        self,
        frame_size=5000,
        background_element=None,
        other_elements=None,
        elements_frequency=None,
        attempts_at_elements=1,
    ):
        self.frame_size = frame_size
        self.background_element = background_element
        self.other_elements = other_elements
        self.elements_frequency = elements_frequency
        self.attempts_at_elements = attempts_at_elements

    def generate_fov(self):
        """Generate a FOV by sampling from the distribution"""
        bg = self.background_element().generate()
        realized_elements = [bg]

        # how many times to try to add an element for each element type
        for n, element in enumerate(self.other_elements):
            if isinstance(self.attempts_at_elements, list) or isinstance(
                self.attempts_at_elements, tuple
            ):
                attempts = int(self.attempts_at_elements[n])
            else:
                attempts = int(self.attempts_at_elements)
            for _ in range(attempts):
                # Generate the element with a certain frequency
                if np.random.rand() <= self.elements_frequency[n]:
                    # Call generate for the element
                    el = element().generate(context_knowledge=realized_elements)
                    # remove cells that are outsite of the field of view
                    el.remove_cells(bg.is_outside(el.cell_centroids))
                    # consider each elements already placed
                    for k in realized_elements:
                        # make room for the new element
                        k.remove_cells(el.is_inside(k.cell_centroids))
                    # add the element to the list of realized elements
                    realized_elements.append(el)

        # combine the results to get a combined field of view
        list_of_cents = [i.cell_centroids for i in realized_elements]
        list_of_probs = [i.cell_probabilities for i in realized_elements]

        # Find the master n_cell_types from all elements
        n_types = 0
        for p in list_of_probs:
            n_types = max(n_types, p.shape[1])

        # Filter out elements with 0 cells, and pad elements with 0 columns
        final_cents = []
        final_probs = []
        for C, P in zip(list_of_cents, list_of_probs):
            if C.shape[0] == 0:
                continue  # Skip elements with no cells

            if P.shape[1] == 0:
                # This has cells but no rules. Pad with zeros.
                final_probs.append(np.zeros((C.shape[0], n_types)))
            else:
                # This has cells and rules.
                final_probs.append(P)
            
            # Add its centroids
            final_cents.append(C)

        if not final_cents:
            # All elements were empty. Create an empty FOV.
            cell_centroids = np.empty((0, 2))
            cell_probabilities = np.empty((0, n_types)) 
        else:
            cell_centroids = np.row_stack(final_cents)
            cell_probabilities = np.row_stack(final_probs)

        fov = FOV(cell_centroids, cell_probabilities)
        fov.realization()
        return fov


class FOV:
    """A field of view with cells and their properties"""

    def __init__(self, cell_centroids, cell_probabilities):
        self.cell_centroids = cell_centroids
        self.cell_probabilities = cell_probabilities
        self.rng = np.random.default_rng()
        self.class_instance_one_hot = None

    def realization(self):
        """Realize the cell types by sampling from the probabilities"""
        self.class_instance_one_hot = self.rng.multinomial(
            n=1, pvals=self.cell_probabilities
        )

    @property
    def class_instance(self):
        return np.argmax(self.class_instance_one_hot, axis=1)

    def make_pandas_df(self):
        """Make a pandas dataframe with the cell properties"""
        data = (
            [*self.cell_centroids.T]
            + [*self.class_instance[:, None].T]
            + [*self.class_instance_one_hot.T]
            + [*self.cell_probabilities.T]
        )
        col_names = (
            [f"X", f"Y", "Class ID"]
            + [f"Class {i}" for i in range(self.class_instance_one_hot.shape[1])]
            + [f"ProbClass{i}" for i in range(self.cell_probabilities.shape[1])]
        )

        try:
            data.append(self.cell_minor_axis)
            col_names.append("Minor Axis")
            data.append(self.cell_major_axis)
            col_names.append("Major Axis")
            data.append(self.cell_rotation)
            col_names.append("Rotation")
            data.append(self.cell_rna_concentration)
            col_names.append("RNA Concentration")
            data.append(np.array([str(i) for i in self.cell_colors]))  # "[1, 0, 0, 1]"
            col_names.append("Color_string")
        except:
            pass
        # prepare different datatypes for the series
        series_dict = {
            col_names[i]: pd.Series(data[i], dtype=data[i].dtype.str)
            for i in range(len(data))
        }
        return pd.DataFrame(series_dict)


class CellTypesProperties:
    """Object that defines the properties of the cell types"""

    def __init__(
        self,
        n_cell_types,
        colordict=None,
        gene_colordict=None,
        name_dict=None,
        sizes=13,
        size_variation=1,
        anisotropy=0.85,
        anisotropy_variation=0.05,
        relative_rna_concentration=1.0,
        rna_concentration_variation=0.05,
    ):
        self.n_cell_types = n_cell_types
        if colordict is None:
            self.colordict = {i: plt.cm.tab20(i) for i in range(n_cell_types)}
        elif isinstance(colordict, matplotlib.colors.Colormap):
            self.colordict = {
                i: colordict(float(i) / n_cell_types) for i in range(n_cell_types)
            }
        elif isinstance(colordict, dict):
            self.colordict = colordict
        else:
            raise ValueError("colordict must be a dictionary or a colormap")

        if gene_colordict is None:
            self.gene_colordict = {i: plt.cm.tab20(i / 200.0) for i in range(200)}
        elif isinstance(gene_colordict, dict):
            self.gene_colordict = gene_colordict
        else:
            raise ValueError("gene_colordict must be a dictionary or a colormap")

        if name_dict is None:
            self.name_dict = {i: f"Type {i+1}" for i in range(self.n_cell_types)}

        # --- START OF FIX ---
        # Check for scalar int OR float, not just int
        if isinstance(sizes, (int, float)):
            self.sizes = np.ones(n_cell_types) * sizes
        else:
            self.sizes = sizes

        if isinstance(size_variation, (int, float)):
            self.size_variation = np.ones(n_cell_types) * size_variation
        else:
            self.size_variation = size_variation

        if isinstance(anisotropy, (int, float)):
            self.anisotropy = np.ones(n_cell_types) * anisotropy
        else:
            self.anisotropy = anisotropy

        if isinstance(anisotropy_variation, (int, float)):
            self.anisotropy_variation = np.ones(n_cell_types) * anisotropy_variation
        else:
            self.anisotropy_variation = anisotropy_variation

        if isinstance(relative_rna_concentration, (int, float)):
            self.relative_rna_concentration = (
                np.ones(n_cell_types) * relative_rna_concentration
            )
        else:
            self.relative_rna_concentration = relative_rna_concentration

        if isinstance(rna_concentration_variation, (int, float)):
            self.rna_concentration_variation = (
                np.ones(n_cell_types) * rna_concentration_variation
            )
        else:
            self.rna_concentration_variation = rna_concentration_variation
        # --- END OF FIX ---

    def apply(self, fov):
        """Apply the properties to a FOV"""
        scell_sizes = fov.class_instance_one_hot @ self.sizes
        scell_anisotropy = fov.class_instance_one_hot @ self.anisotropy
        scell_size_variation = fov.class_instance_one_hot @ self.size_variation
        scell_relative_rna_concentration = (
            fov.class_instance_one_hot @ self.relative_rna_concentration
        )
        scell_rna_concentration_variation = (
            fov.class_instance_one_hot @ self.rna_concentration_variation
        )
        scell_anisotropy_variation = (
            fov.class_instance_one_hot @ self.anisotropy_variation
        )
        # this is just random because I don't have time now
        scell_rotation = np.random.uniform(
            0, 2 * np.pi, fov.class_instance_one_hot.shape[0]
        )

        # Generate random minor axis lengths, major axis lengths
        scell_radius = np.clip(
            np.random.normal(scell_sizes, scell_size_variation),
            scell_sizes * 0.2,
            scell_sizes * 3,
        )
        scell_anisotropy_realized = np.clip(
            np.random.normal(scell_anisotropy, scell_anisotropy_variation),
            0.2,
            2 * scell_anisotropy,
        )
        scell_concentration_realized = np.random.normal(
            scell_relative_rna_concentration, scell_rna_concentration_variation
        )

        fov.cell_minor_axis = scell_radius * scell_anisotropy_realized
        fov.cell_major_axis = scell_radius * (2 - scell_anisotropy_realized)
        fov.cell_rotation = scell_rotation
        fov.cell_rna_concentration = scell_concentration_realized
        fov.cell_colors = [self.colordict[i] for i in fov.class_instance]


class TissueCellTypes:
    """A tissue with cell types and gene expression patterns"""

    def __init__(self, gene_expression_by_type=None):
        self.gene_expression_by_type = gene_expression_by_type
        self.concentration = None

    def generate_types_and_markers(
        self,
        n_genes,
        n_cell_types,
        expected_level=8.0,
        expected_std_level=3.0,
        concentration=0.90,
    ):
        self.expected_level = expected_level
        self.concentration = concentration
        tmp = np.round(
            np.random.dirichlet(
                np.ones(n_cell_types) * (1 - self.concentration), n_genes
            ),
            int(2.5 * np.log10(n_genes)),
        )
        self.fuzzy_sparsity_pattern = tmp / tmp.sum(axis=0)

        self.scale_of_the_gene = intuitive_rand_lognormal(
            expected_level, expected_std_level, n_genes
        )
        self.gene_expression_by_type = (
            self.scale_of_the_gene[:, None] * self.fuzzy_sparsity_pattern
        )

        self.colordict = {
            i: plt.cm.turbo(float(i) / n_cell_types) for i in range(n_cell_types)
        }

        self.gene_colordict = {
            i: plt.cm.turbo(float(i) / n_genes) for i in range(n_genes)
        }

        ixs = np.argsort(self.gene_expression_by_type.argmax(1))
        self.gene_expression_by_type = self.gene_expression_by_type[ixs, :]

        return self.gene_expression_by_type

    @property
    def n_cell_types(self):
        return self.gene_expression_by_type.shape[1]

    @property
    def n_genes(self):
        return self.gene_expression_by_type.shape[0]

    @property
    def cell_type_names(self):
        return np.array([f"Type {i}" for i in range(self.n_cell_types)])

    @property
    def gene_names(self):
        return np.array([f"Gene {i}" for i in range(self.n_genes)])

    def make_pandas_df(self):
        return pd.DataFrame(
            self.gene_expression_by_type,
            columns=self.cell_type_names,
            index=self.gene_names,
        )


class TransferFunctionBase:
    def __init__(self, **kwargs):
        self.params = kwargs

    def transform(self):
        raise NotImplementedError("This is an abstract method")


class IdentityTransfer(TransferFunctionBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, x):
        return x


class AffineNonNegTransfer(TransferFunctionBase):
    def __init__(self, scales=None, offsets=None, scales_std=None, offsets_std=None):
        """An affine transfer function

        Parameters
        ----------
        scales : np.ndarray
            The scales for each gene
        offsets : np.ndarray
            The offsets for each gene
        scales_std : float
            The standard deviation for the scales
        offsets_std : float
            The standard deviation for the offsets

        Description of what the parameters are used to and what the function does

        scales will be randomly drawn from a lognormal distribution with mean and std = scales, scales_std
        offsets will be randomly drawn from a normal distribution with mean and std = offsets, offsets_std

        The function will return affine transformation that is clipped to be non-negative
        """
        super().__init__(
            scales=scales,
            offsets=offsets,
            scales_std=scales_std,
            offsets_std=offsets_std,
        )
        self.scales = None
        self.offsets = None

    def transform(self, x):
        n = x.shape[0]
        if self.scales is not None:
            pass
        elif self.params["scales"] is None:
            raise ValueError("scales must be provided")
        elif isinstance(self.params["scales"], float) or isinstance(
            self.params["scales"], int
        ):
            self.scales = np.random.lognormal(
                *lognorm_params_to_mean_std(
                    self.params["scales"], self.params["scales_std"]
                ),
                n,
            )
        else:
            self.scales = self.params["scales"]

        if self.offsets is not None:
            pass
        elif self.params["offsets"] is None:
            raise ValueError("offsets must be provided")
        elif isinstance(self.params["offsets"], float) or isinstance(
            self.params["offsets"], int
        ):
            self.offsets = np.random.normal(
                self.params["offsets"], self.params["offsets_std"], n
            )
        else:
            self.offsets = self.params["offsets"]

        return np.maximum(self.scales[:, None] * x + self.offsets[:, None], 1e-6)


class HybISS_Setup:
    """A setup for the HybISS experiment with a tissue and a set of parameters for the experiment"""

    def __init__(
        self,
        tissue,
        genes_sensitivities=1.0,
        genes_sensitivities_variation=0.3,
        transfer_function=IdentityTransfer(),
    ):
        self.tissue = tissue
        self.transfer_function = transfer_function
        self.raw_M = tissue.gene_expression_by_type  # shape = (gene, types)
        self.M = self.transfer_function.transform(self.raw_M)
        self.rng = np.random.default_rng()
        self.cellxgene_counts = None
        if isinstance(genes_sensitivities_variation, float):
            self.genes_sensitivities_variation = (
                np.ones(self.M.shape[0]) * genes_sensitivities_variation
            )
        else:
            self.genes_sensitivities_variation = genes_sensitivities_variation

        if isinstance(genes_sensitivities, float):
            tmp = np.ones(self.M.shape[0]) * genes_sensitivities
            self.genes_sensitivities = np.random.lognormal(
                *lognorm_params_to_mean_std(tmp, self.genes_sensitivities_variation)
            )
        else:
            self.genes_sensitivities = genes_sensitivities

    def measure_gene_expression(self, fov):
        """Measure gene expression in cells with HybISS"""
        cellxgene_expectation = (
            fov.cell_rna_concentration[:, None]
            * (fov.class_instance_one_hot @ self.M.T)
            * self.genes_sensitivities
        )
        self.cellxgene_counts = self.rng.poisson(cellxgene_expectation)
        self.cellxtotal_counts = self.cellxgene_counts.sum(axis=1)
        return self.cellxgene_counts

    def observe_dots(self, fov):
        """Observe dots in cells with HybISS"""
        
        # --- FIX: Unconditionally measure the new fov and clear old state ---
        self.measure_gene_expression(fov)
        
        self.dot_belongsto_by_cells = []
        self.dot_isgene_by_cells = []
        self.dot_xs_by_cells = []
        self.dot_ys_by_cells = []
        # --- End Fix ---

        for i in range(self.cellxgene_counts.shape[0]):
            xs, ys = generate_points_asin_cell(
                fov.cell_centroids[i],
                1.0,
                self.cellxtotal_counts[i],
                fov.cell_major_axis[i],
                fov.cell_minor_axis[i],
                fov.cell_rotation[i],
            ).T
            self.dot_xs_by_cells.append(xs)
            self.dot_ys_by_cells.append(ys)
            tmp_is = []
            tmp_js = []
            for j in range(self.cellxgene_counts.shape[1]):
                for _ in range(self.cellxgene_counts[i, j]):
                    tmp_js.append(j)
                    tmp_is.append(i)
            self.dot_belongsto_by_cells.append(tmp_is)
            self.dot_isgene_by_cells.append(tmp_js)

    def make_pandas_df(self):
        """Make a pandas dataframe with the dot properties"""
        cell_ixs = np.concatenate(self.dot_belongsto_by_cells).astype(int)
        gene_ixs = np.concatenate(self.dot_isgene_by_cells).astype(int)
        return pd.DataFrame(
            {
                "x": np.concatenate(self.dot_xs_by_cells),
                "y": np.concatenate(self.dot_ys_by_cells),
                "gene": pd.Series(self.tissue.gene_names[gene_ixs], dtype=str),
                "cell": pd.Series(cell_ixs, dtype=int),
            }
        )


def generate_dataset(outfolder, fovd, hybISS, tissue, cell_props, n_samples=100):
    """Generate a dataset of FOVs with ground truth and data

    The function generates a dataset of FOVs with ground truth and data
    and saves them in a folder with the following structure:

    outfolder
    ├── data
    ├── ground_truths
    └── parameters
        └── cell_types.csv

    Parameters
    ----------
    outfolder : str
        The folder where to save the dataset
    fovd : FOVDistribution
        The distribution of FOVs to sample from
    hybISS : HybISS_Setup
        The setup for the HybISS experiment
    tissue : TissueCellTypes
        The tissue with cell types and gene expression patterns
    cell_props : CellTypesProperties
        The properties of the cell types
    n_samples : int
        The number of samples to generate
    """
    # if outfolder does not exist, create it

    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    # create 3 subfolders data, ground_truths, parameters
    if not os.path.exists(outfolder + "/data"):
        os.makedirs(outfolder + "/data")
    if not os.path.exists(outfolder + "/ground_truths"):
        os.makedirs(outfolder + "/ground_truths")
    if not os.path.exists(outfolder + "/parameters"):
        os.makedirs(outfolder + "/parameters")

    # save tissue parameters
    cell_type_df = tissue.make_pandas_df()
    cell_type_df.to_csv(outfolder + "/parameters/cell_types.csv")

    for i in tqdm.tqdm(range(n_samples)):
        fov = fovd.generate_fov()
        cell_props.apply(fov)
        fov_name = f"fov_{i+1:04}"
        # make a pandas dataframe for the ground truth
        ground_truth = fov.make_pandas_df()
        cell_centers = ground_truth.loc[:, ["X", "Y"]]
        # save ground truth
        # zero pad the file name to 4 digits
        ground_truth.to_csv(outfolder + f"/ground_truths/cells_FOV{i+1:04}.csv")
        # save cell centers as data
        cell_centers.to_csv(outfolder + f"/data/cell_centroids_FOV{i+1:04}.csv")

        # Measure gene expression in cells with HybISS
        hybISS.measure_gene_expression(fov)

        # Observe dots
        hybISS.observe_dots(fov)
        ground_truth_dots = hybISS.make_pandas_df()
        data = ground_truth_dots.loc[:, ["x", "y", "gene"]]
        # save data
        ground_truth_dots.to_csv(outfolder + f"/ground_truths/dots_FOV{i+1:04}.csv")
        data.to_csv(outfolder + f"/data/dots_FOV{i+1:04}.csv")

    tissue_hybISS = copy.deepcopy(tissue)
    tissue_hybISS.gene_expression_by_type = hybISS.M
    cell_type_df_hybISS = tissue_hybISS.make_pandas_df()
    cell_type_df_hybISS.to_csv(outfolder + "/parameters/cell_types_hybISS.csv")


def load_data(outfolder, return_transformed=False):
    """Load all the data in a folder

    The function assumes the data was generated calling generate_dataset
    but does not assume any of the parameters so it needs to check for how many files are in the folder

    Parameters
    ----------
    outfolder : str
        The folder where the dataset is saved

    Returns
    -------
    data : dict
        A dictionary with the data
    """

    data = defaultdict(OrderedDict)

    for file in sorted(glob.glob(outfolder + "/ground_truths/*.csv")):
        filename = os.path.splitext(os.path.basename(file))[0]
        fov_name = filename.split("_")[-1]
        if "cells" in file:
            data["ground_truth"][fov_name] = pd.read_csv(file, index_col=0)
        elif "dots" in file:
            data["ground_truth_dots"][fov_name] = pd.read_csv(file, index_col=0)

    for file in sorted(glob.glob(outfolder + "/data/*.csv")):
        filename = os.path.splitext(os.path.basename(file))[0]
        fov_name = filename.split("_")[-1]
        if "cell_centroids" in file:
            data["cell_centroids"][fov_name] = pd.read_csv(file, index_col=0)
        elif "dots" in file:
            data["dots"][fov_name] = pd.read_csv(file, index_col=0)

    cell_types_info = pd.read_csv(outfolder + "/parameters/cell_types.csv", index_col=0)

    if return_transformed:
        cell_types_info_hybISS = pd.read_csv(
            outfolder + "/parameters/cell_types_hybISS.csv", index_col=0
        )
        return data, cell_types_info, cell_types_info_hybISS
    else:
        return data, cell_types_info


class DeterministicNeighborAssignment(CellTypeRuleBase):
    """Rule that assigns a single cell type to all cells"""

    def __init__(
        self,
        n_cell_types,
        close_to=None,
        becomes=None,
        threshold_probdist=0.3,
        allow_selftype_change=False,
        mix=0.9,
        nneigh=2,
    ):
        super().__init__(n_cell_types=n_cell_types)
        if close_to is None:
            self.close_to = np.random.randint(0, n_cell_types)
        else:
            self.close_to = close_to
        if becomes is None:
            self.becomes = np.random.randint(0, n_cell_types)
        else:
            self.becomes = becomes
        self.close_to_onehot = np.zeros(self.n_cell_types)
        self.close_to_onehot[self.close_to] = 1
        self.becomes_onehot = np.zeros(self.n_cell_types)
        self.becomes_onehot[self.becomes] = 1
        self.threshold = threshold_probdist
        self.allow_selftype_change = allow_selftype_change
        self.mix = mix
        self.nneigh = nneigh

    def apply(self, cell_centroids, current_probs):
        """Assign a single cell type for each cell centroid"""
        # one hot encode close_to and becomes
        new_probs = np.copy(current_probs)
        kd = KDTree(cell_centroids)
        dist_prob = np.sqrt(np.sum((current_probs - self.close_to_onehot) ** 2, 1))
        type_of_focus = dist_prob < self.threshold
        # find the first nearest neighbor not self
        dist_, ind_ = kd.query(cell_centroids, k=1 + self.nneigh)
        for i in range(self.nneigh):
            dist, ind = dist_[:, 1 + i], ind_[:, 1 + i]
            if not self.allow_selftype_change:
                nearestneigh_is_focustype = type_of_focus[ind]
                bool_ = type_of_focus & ~nearestneigh_is_focustype
            else:
                bool_ = type_of_focus
            ixes = ind[bool_]
            new_probs[ixes, :] = (1 - self.mix) * new_probs[
                ixes, :
            ] + self.mix * self.becomes_onehot

            # Normalize the probabilities (as operation above are not guaranteed in the simplex)
            cell_probs = np.maximum(new_probs, 1e-12)
            cell_probs = cell_probs / (cell_probs.sum(axis=1)[:, np.newaxis] + 1e-8)

        return cell_probs


class DummyRule(CellTypeRuleBase):
    def __init__(self, n_cell_types, probs_to_copy=None):
        super().__init__(n_cell_types)
        self.probs_to_copy = probs_to_copy

    def apply(self, cell_centroids, current_probs=None, **kwargs):
        return self.probs_to_copy

    def adapt_rule_to_element(self, element):
        return self


class FrameWideUpdater(FrameWideElement):
    """It clones the underlying elements and apply its rules

    IMPORANT NOTE!!!
    For now it has a bug so that it only works once
    """

    def __init__(self, rules):
        super().__init__(rules=rules)

    def apply_rules(self, rules):
        probs = None
        for rule in rules:  # <-- FIX: Must iterate over the 'rules' argument, not 'self.rules'
            probs = rule.apply(self.cell_centroids, current_probs=probs)

        # If no rules were applied (e.g., rules=[]), probs is still None.
        # We must return an array, not None, for slicing to work.
        if probs is None:
            n_cells = self.cell_centroids.shape[0]
            n_types = 0  # Default
            if rules:  # <-- FIX: Must check the 'rules' argument, not 'self.rules'
                # Attempt to get n_cell_types from the first rule if it exists
                # We add a fallback in case rules[0] doesn't have n_cell_types
                try:
                    n_types = rules[0].n_cell_types
                except (IndexError, AttributeError):
                    n_types = 0 # Default to 0 if no rules or rule has no n_cell_types
            
            # Return an array with the correct dimensions (e.g., shape (N, 0))
            probs = np.empty((n_cells, n_types))

        return probs

    def generate(self, **kwargs):
        # make a copy of self reinitialize the element
        other = copy.deepcopy(self)
        other.polygon = None
        other.cell_centroids = None
        other._class_instance_one_hot = None
        other.rules = other.original_rules

        realized_elements = kwargs["context_knowledge"]  # key argument
        bg = realized_elements[0]
        other.polygon = bg.polygon
        other.cell_centroids = np.row_stack(
            [i.cell_centroids for i in realized_elements]
        )
        other.probs_to_update = np.row_stack(
            [i.cell_probabilities for i in realized_elements]
        )
        # update rules using this trick
        tmp = [i.adapt_rule_to_element(other) for i in other.rules]
        other.rules = [
            DummyRule(
                n_cell_types=other.cell_centroids.shape[1],
                probs_to_copy=other.probs_to_update,
            )
        ] + tmp
        other.cell_probabilities = other.apply_rules(other.rules)
        return other

    def is_inside(self, points):
        # It is always inside (hack to remove the rest)
        return np.ones(
            len(points),
            dtype=bool,
        )

    def is_outside(self, points):
        return ~self.is_inside(points)

    def remove_cells(self, bool_ix):
        self.cell_centroids = self.cell_centroids[~bool_ix]
        self.cell_probabilities = self.cell_probabilities[~bool_ix]


class SingleTypeRule(CellTypeRuleBase):
    """Rule that assigns a single cell type to all cells"""

    def __init__(self, n_cell_types, cell_type_ix=None):
        super().__init__(n_cell_types)
        if cell_type_ix is None:
            self.cell_type_ix = np.random.randint(self.n_cell_types)
        else:
            self.cell_type_ix = cell_type_ix

    def apply(self, cell_centroids, current_probs=None, mix=0.9):
        """Assign a single cell type for each cell centroid"""
        new_probs = np.zeros((len(cell_centroids), self.n_cell_types), dtype=float)
        new_probs[:, self.cell_type_ix] = 0.9999999
        if current_probs is None:
            cell_probs = new_probs
        else:
            cell_probs = mix * new_probs + (1 - mix) * current_probs
        cell_probs = np.maximum(cell_probs, 1e-12)

        # Normalize the probabilities (as operation above are not guaranteed in the simplex)
        cell_probs = cell_probs / (cell_probs.sum(axis=1)[:, np.newaxis] + 1e-8)

        return cell_probs


# --- NEW: region + slice composer -------------------------------------------
from dataclasses import dataclass
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union

@dataclass
class RegionSpec:
    name: str
    polygon: Polygon                   # where this region lives (rect or freeform)
    fovdist: FOVDistribution           # region-specific scene generator
    priority: int = 0                  # later regions can overwrite earlier ones in overlaps
    blend_band: float = 0.0            # optional: width (in px) for boundary blending

class TissueSlice:
    """
    Compose multiple region-specific FOVDistributions on a single global canvas,
    then export overlapping FOV tiles that are perfect copies in the overlaps.
    """
    def __init__(self, frame_size: int, regions: list[RegionSpec]):
        self.frame_size = frame_size
        self.regions = sorted(regions, key=lambda r: r.priority)
        self._cells_xy = None
        self._probs = None
        self._class_onehot = None

    def generate_global(self):
        realized = []   # list of dicts with xy, probs
        occupied_mask = None

        for reg in self.regions:
            fov = reg.fovdist.generate_fov()  # <- uses your existing generator
            xy  = fov.cell_centroids
            P   = fov.cell_probabilities

            # clip to region polygon
            inside = np.array([reg.polygon.contains(Point(x,y)) for x,y in xy], bool)
            xy, P = xy[inside], P[inside]

            # optionally blend at boundaries (softly mix with distance to boundary)
            if reg.blend_band > 0:
                bdist = np.array([reg.polygon.boundary.distance(Point(x,y)) for x,y in xy])
                w = np.clip(bdist / reg.blend_band, 0.0, 1.0)[:, None]  # 0 at edge, 1 inside
                # simple inward sharpening: push probs toward argmax near boundaries
                hard = np.eye(P.shape[1])[np.argmax(P, 1)]
                P = w * P + (1.0 - w) * hard

            realized.append({"xy": xy, "P": P})

        # Guard against no cells being generated
        if not realized:
            self._cells_xy = np.empty((0, 2))
            self._probs = np.empty((0, 0))
            self._class_onehot = None
            return self

        # Find the master number of cell types across all regions
        n_types = 0
        for r in realized:
            if r["P"].shape[0] > 0: # Check if there are any cells
                n_types = max(n_types, r["P"].shape[1])
        
        # Stack all xy and (padded) P arrays
        all_xy_list = []
        all_P_list = []
        for r in realized:
            if r["xy"].shape[0] == 0:
                continue # Skip regions that ended up with 0 cells
            
            all_xy_list.append(r["xy"])
            
            # Pad P if necessary
            P = r["P"]
            if P.shape[1] < n_types:
                P_padded = np.zeros((P.shape[0], n_types))
                P_padded[:, :P.shape[1]] = P
                all_P_list.append(P_padded)
            else:
                all_P_list.append(P)

        # Handle case where all regions were empty after filtering
        if not all_xy_list:
            self._cells_xy = np.empty((0, 2))
            self._probs = np.empty((0, n_types))
            self._class_onehot = None
            return self

        all_xy = np.row_stack(all_xy_list)
        all_P = np.row_stack(all_P_list)

        # resolve overlaps by region priority (later wins)
        # concatenate then drop duplicates by keeping last occurrence
        # grid-hash to merge near-identical points (jitter)
        key = np.round(all_xy, 3).view([('', all_xy.dtype)]*2).flatten() # 1e-3 px tolerance

        # Get unique keys and the index of their *last* appearance to respect priority
        unique_keys_reversed, last_indices_from_reversed = np.unique(key[::-1], return_index=True)
        
        # Re-calculate original indices from the reversed array
        keep_idx = (len(key) - 1 - last_indices_from_reversed)
        
        # Sort indices to maintain spatial coherence (optional but good practice)
        keep_idx.sort() 

        self._cells_xy = all_xy[keep_idx]
        self._probs    = all_P[keep_idx]
        self._class_onehot = None  # sample later if you want hard labels
        return self

    # def sample_labels(self, rng=None):
    #     rng = np.random.default_rng() if rng is None else rng
    #     n, k = self._probs.shape
    #     draws = np.array([rng.choice(k, p=p/np.clip(p.sum(),1e-9,None)) for p in self._probs])
    #     self._class_onehot = np.eye(k, dtype=int)[draws]
    #     return self

    def sample_labels(self, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        n, k = self._probs.shape

        # Handle edge case where there are no cell types at all
        if k == 0:
            self._class_onehot = np.empty((n, 0), dtype=int)
            return self
            
        draws = np.zeros(n, dtype=int)
        
        # Create a uniform probability vector to use for all-zero rows
        uniform_probs = np.ones(k) / k
        
        for i, p in enumerate(self._probs):
            p_sum = p.sum()
            
            if p_sum > 1e-8:
                # Normal case: probabilities sum to something
                p_normalized = p / p_sum
            else:
                # Problem case: all-zero probabilities. Assign a uniform distribution.
                p_normalized = uniform_probs
            
            # This try-except block adds robustness against tiny fp errors
            try:
                draws[i] = rng.choice(k, p=p_normalized)
            except ValueError:
                # Fallback for rare cases where sum is e.g. 0.999999999999
                draws[i] = np.argmax(p_normalized) # Just pick the max

        self._class_onehot = np.eye(k, dtype=int)[draws]
        return self

    def tile_into_fovs(self, fov_size: int, overlap_frac: float = 0.10):
        """
        Returns a list of per-FOV selections (indices into the global arrays).
        Adjacent FOVs share exact cells in the overlapped margins.
        """
        step = int(round(fov_size * (1.0 - overlap_frac)))
        xs = list(range(0, self.frame_size - fov_size + 1, step))
        ys = list(range(0, self.frame_size - fov_size + 1, step))
        tiles = []
        X, Y = self._cells_xy[:,0], self._cells_xy[:,1]
        for j, y0 in enumerate(ys):
            for i, x0 in enumerate(xs):
                x1, y1 = x0 + fov_size, y0 + fov_size
                keep = (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
                tiles.append({
                    "ij": (i, j),
                    "bbox": (x0, y0, x1, y1),
                    "idx": np.where(keep)[0]
                })
        return tiles
"""
Example Usage:

N_CELL_TYPES = 10
N_GENES = 12
FRAME = 1000

# Create a tissue with 10 cell types
tissue = TissueCellTypes()
tissue.generate_types_and_markers(N_GENES, N_CELL_TYPES, expected_level=10, expected_std_level=2)
cell_type_df = tissue.make_pandas_df()

# Create a set of properties for the cell types
cell_props = CellTypesProperties(n_cell_types=N_CELL_TYPES)

# Create a set of rules for the generation of the histological elements

# Background element
ruleset1 = [ProbabilityNodeFieldRule(N_CELL_TYPES, 3, 0.1)]
BackGroundPrototype1 = lambda: FrameWideElement(frame_size=FRAME, n_vertices=4,
                                                scale=1000, tipical_cell_spacing=16, rules=ruleset1)

# Large histological element
ruleset2 = [RandomCellTypeRule(N_CELL_TYPES), ProbabilityNodeFieldRule(N_CELL_TYPES, 3, 0.05)]
HistologicalElementType1 = lambda: HistologicalElement(frame_size=FRAME, n_vertices=20,
                                                       scale=600, tipical_cell_spacing=14, rules=ruleset2)

# Small histological element
ruleset3 = [RandomCellTypeRule(N_CELL_TYPES)]
HistologicalElementType2 = lambda: HistologicalElement(frame_size=FRAME, n_vertices=(5,8),
                                                       scale=200, tipical_cell_spacing=12, rules=ruleset3)

# Create a FOV distribution to sample from
fovd = FOVDistribution(frame_size=FRAME,
                background_element=BackGroundPrototype1,
                other_elements=[HistologicalElementType1, HistologicalElementType2],
                elements_frequency=(0.3, 0.9))

# Generate an experiment
hybISS = HybISS_Setup(tissue.gene_expression_by_type,
                      genes_sensitivities=1.0, genes_sensitivities_variation=0.3)

# Generate a FOV
fov = fovd.generate_fov()


# Apply the cell properties defined above
cell_props.apply(fov)
ground_truth = fov.make_pandas_df()

# Measure gene expression in cells with HybISS
hybISS.measure_gene_expression(fov)

# Observe dots
hybISS.observe_dots(fov)

# make a pandas dataframe for the counts
df = hybISS.make_pandas_df()
"""