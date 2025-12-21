"""Effect controller for managing simulation effects."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Set
import numpy as np


@dataclass
class EffectController:
    """Controller for enabling/disabling and tuning simulation effects.

    The EffectController manages which effects are active in a simulation
    and their relative strengths. This allows for controlled experiments
    where specific sources of variation can be isolated or combined.

    Parameters
    ----------
    enabled_effects : Set[str], optional
        Set of enabled effect names. Default enables all.
    effect_magnitudes : Dict[str, float], optional
        Magnitude multipliers for each effect (0-1 scale).

    Attributes
    ----------
    KNOWN_EFFECTS : set
        Set of all recognized effect names.

    Examples
    --------
    >>> controller = EffectController()
    >>> controller.disable("batch_effect")
    >>> controller.set_magnitude("dropout", 0.5)  # Half strength
    >>> if controller.is_enabled("technical_noise"):
    ...     # Apply technical noise
    """

    KNOWN_EFFECTS: Set[str] = field(
        default_factory=lambda: {
            "batch_effect",
            "technical_noise",
            "background_noise",
            "dropout",
            "vignetting",
            "amplification_noise",
            "cell_size_variation",
            "spatial_noise",
            "expression_noise",
        },
        repr=False
    )

    enabled_effects: Set[str] = field(default_factory=set)
    effect_magnitudes: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize with all effects enabled by default."""
        if not self.enabled_effects:
            self.enabled_effects = self.KNOWN_EFFECTS.copy()

        # Default magnitudes are 1.0
        for effect in self.KNOWN_EFFECTS:
            if effect not in self.effect_magnitudes:
                self.effect_magnitudes[effect] = 1.0

    def enable(self, effect: str) -> "EffectController":
        """Enable an effect.

        Parameters
        ----------
        effect : str
            Effect name to enable.

        Returns
        -------
        EffectController
            Self, for method chaining.
        """
        if effect not in self.KNOWN_EFFECTS:
            raise ValueError(f"Unknown effect: {effect}. Known effects: {self.KNOWN_EFFECTS}")
        self.enabled_effects.add(effect)
        return self

    def disable(self, effect: str) -> "EffectController":
        """Disable an effect.

        Parameters
        ----------
        effect : str
            Effect name to disable.

        Returns
        -------
        EffectController
            Self, for method chaining.
        """
        self.enabled_effects.discard(effect)
        return self

    def enable_all(self) -> "EffectController":
        """Enable all effects.

        Returns
        -------
        EffectController
            Self, for method chaining.
        """
        self.enabled_effects = self.KNOWN_EFFECTS.copy()
        return self

    def disable_all(self) -> "EffectController":
        """Disable all effects.

        Returns
        -------
        EffectController
            Self, for method chaining.
        """
        self.enabled_effects = set()
        return self

    def enable_only(self, effects: Set[str]) -> "EffectController":
        """Enable only the specified effects.

        Parameters
        ----------
        effects : set
            Set of effect names to enable (all others disabled).

        Returns
        -------
        EffectController
            Self, for method chaining.
        """
        for effect in effects:
            if effect not in self.KNOWN_EFFECTS:
                raise ValueError(f"Unknown effect: {effect}")
        self.enabled_effects = effects.copy()
        return self

    def is_enabled(self, effect: str) -> bool:
        """Check if an effect is enabled.

        Parameters
        ----------
        effect : str
            Effect name to check.

        Returns
        -------
        bool
            True if effect is enabled.
        """
        return effect in self.enabled_effects

    def set_magnitude(self, effect: str, magnitude: float) -> "EffectController":
        """Set the magnitude of an effect.

        Parameters
        ----------
        effect : str
            Effect name.
        magnitude : float
            Magnitude multiplier (0=off, 1=full strength).

        Returns
        -------
        EffectController
            Self, for method chaining.
        """
        if effect not in self.KNOWN_EFFECTS:
            raise ValueError(f"Unknown effect: {effect}")
        if not 0 <= magnitude <= 2:
            raise ValueError("magnitude should be in [0, 2]")
        self.effect_magnitudes[effect] = magnitude
        return self

    def get_magnitude(self, effect: str) -> float:
        """Get the magnitude of an effect.

        Parameters
        ----------
        effect : str
            Effect name.

        Returns
        -------
        float
            Magnitude multiplier.
        """
        if not self.is_enabled(effect):
            return 0.0
        return self.effect_magnitudes.get(effect, 1.0)

    def get_effective_magnitude(self, effect: str) -> float:
        """Get effective magnitude considering enabled state.

        Parameters
        ----------
        effect : str
            Effect name.

        Returns
        -------
        float
            0.0 if disabled, otherwise the set magnitude.
        """
        if not self.is_enabled(effect):
            return 0.0
        return self.effect_magnitudes.get(effect, 1.0)

    @property
    def active_effects(self) -> Set[str]:
        """Set[str]: Currently enabled effects."""
        return self.enabled_effects.copy()

    @property
    def summary(self) -> Dict[str, Any]:
        """dict: Summary of effect states and magnitudes."""
        return {
            effect: {
                "enabled": self.is_enabled(effect),
                "magnitude": self.get_effective_magnitude(effect),
            }
            for effect in self.KNOWN_EFFECTS
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "enabled_effects": list(self.enabled_effects),
            "effect_magnitudes": self.effect_magnitudes.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffectController":
        """Create from dictionary."""
        return cls(
            enabled_effects=set(data.get("enabled_effects", [])),
            effect_magnitudes=data.get("effect_magnitudes", {}).copy(),
        )

    @classmethod
    def clean_simulation(cls) -> "EffectController":
        """Create a controller with all effects disabled (clean simulation).

        Returns
        -------
        EffectController
            Controller with no effects enabled.
        """
        return cls(enabled_effects=set())

    @classmethod
    def realistic_simulation(cls) -> "EffectController":
        """Create a controller with realistic effect settings.

        Returns
        -------
        EffectController
            Controller with all effects enabled at moderate levels.
        """
        controller = cls()
        controller.set_magnitude("batch_effect", 0.5)
        controller.set_magnitude("dropout", 0.7)
        controller.set_magnitude("background_noise", 0.3)
        return controller
