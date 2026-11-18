"""GPS L1 C/A spoofing detection on TEXBAT via correlation-based SQM and machine learning."""

from .classifiers import (
    ModelEvaluation,
    compute_feature_importance,
    evaluate_leave_one_attack_out,
    evaluate_stratified_split,
    evaluate_temporal_block_split,
    split_temporal_blocks,
)
from .data_loader import (
    FEATURE_COLUMNS,
    TEXBAT_SCENARIOS,
    TexbatScenarioSpec,
    build_texbat_gnsssdr_dataset,
    extract_gnsssdr_scenario_windows,
    load_or_synthesize_raw_iq,
    load_texbat_feature_dataset,
)
from .signal_processing import (
    DEFAULT_FS_HZ,
    GPS_CA_CODE_LENGTH,
    GPS_CHIP_RATE_HZ,
    AcquisitionResult,
    SpectralMetrics,
    acquire_prn_pcps,
    compute_psd_and_flatness,
    generate_ca_code,
    load_iq_from_bin,
    sample_ca_code,
    scan_visible_satellites,
)
from .sqm_features import (
    RCPResult,
    calculate_rcp,
    extract_multichannel_iq_features,
    extract_waveform_sqm_features,
)
from .synthetic_texbat import generate_synthetic_iq_scenario

__all__ = [
    "AcquisitionResult",
    "DEFAULT_FS_HZ",
    "FEATURE_COLUMNS",
    "GPS_CA_CODE_LENGTH",
    "GPS_CHIP_RATE_HZ",
    "ModelEvaluation",
    "RCPResult",
    "SpectralMetrics",
    "TEXBAT_SCENARIOS",
    "TexbatScenarioSpec",
    "acquire_prn_pcps",
    "build_texbat_gnsssdr_dataset",
    "calculate_rcp",
    "compute_feature_importance",
    "compute_psd_and_flatness",
    "evaluate_leave_one_attack_out",
    "evaluate_stratified_split",
    "evaluate_temporal_block_split",
    "extract_gnsssdr_scenario_windows",
    "extract_multichannel_iq_features",
    "extract_waveform_sqm_features",
    "generate_ca_code",
    "generate_synthetic_iq_scenario",
    "load_iq_from_bin",
    "load_or_synthesize_raw_iq",
    "load_texbat_feature_dataset",
    "sample_ca_code",
    "scan_visible_satellites",
    "split_temporal_blocks",
]
