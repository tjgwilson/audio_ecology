from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl

from audio_ecology.analysis.birdnet import (
    BIRDNET_BACKEND,
    birdnet_week_from_timestamp,
    get_birdnet_checkpoint_store,
    get_birdnet_output_dir,
    load_birdnet_model,
    normalise_birdnet_predictions,
    run_birdnet_analysis,
    run_birdnet_predictions,
)
from audio_ecology.config import (
    BirdNETConfig,
    LocationConfig,
    OutputConfig,
    PipelineConfig,
)


class FakeBirdNETModel:
    """Small fake BirdNET model for tests."""

    def __init__(self) -> None:
        self.audio_paths: list[str] = []
        self.predict_kwargs: list[dict[str, object]] = []

    def predict(self, audio_path: str, **kwargs: object) -> pl.DataFrame:
        self.audio_paths.append(audio_path)
        self.predict_kwargs.append(kwargs)
        return pl.DataFrame(
            [
                {
                    'input': audio_path,
                    'start_time': '00:00:03.00',
                    'end_time': '00:00:06.00',
                    'species_name': 'Erithacus rubecula_European Robin',
                    'confidence': 0.876,
                },
                {
                    'input': audio_path,
                    'start_time': '00:00:06.00',
                    'end_time': '00:00:09.00',
                    'species_name': 'Turdus merula_Eurasian Blackbird',
                    'confidence': 0.1,
                },
            ]
        )


class FakeGeoPredictionResult:
    """Small fake BirdNET geo prediction result for tests."""

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                {
                    'species_name': 'Erithacus rubecula_European Robin',
                    'confidence': 0.812,
                },
                {
                    'species_name': 'Turdus merula_Eurasian Blackbird',
                    'confidence': 0.746,
                },
            ]
        )

    def to_set(self) -> set[str]:
        return {
            'Erithacus rubecula_European Robin',
            'Turdus merula_Eurasian Blackbird',
        }


class FakeBirdNETGeoModel:
    """Small fake BirdNET geo model for tests."""

    def __init__(self) -> None:
        self.predict_calls: list[tuple[float, float, int | None, float]] = []

    def predict(
        self,
        latitude: float,
        longitude: float,
        *,
        week: int | None = None,
        min_confidence: float = 0.03,
    ) -> FakeGeoPredictionResult:
        self.predict_calls.append((latitude, longitude, week, min_confidence))
        return FakeGeoPredictionResult()


def make_config(tmp_path: Path) -> PipelineConfig:
    """Create a simple BirdNET-enabled config."""
    return PipelineConfig(
        project_root=tmp_path,
        input_dir=tmp_path / 'raw',
        output_dir=tmp_path / 'processed',
        site_name='Test Site',
        fallback_location=LocationConfig(
            latitude=51.5,
            longitude=-2.1,
        ),
        devices={},
        birdnet=BirdNETConfig(
            output_dir=tmp_path / 'processed' / 'birdnet',
            model_version='2.4',
            model_backend='tf',
            min_confidence=0.4,
            batch_size=2,
            sensitivity=1.2,
        ),
    )


def make_inventory_df(tmp_path: Path) -> pl.DataFrame:
    """Create a small inventory DataFrame with AudioMoth metadata."""
    return pl.DataFrame(
        [
            {
                'file_path': str(
                    tmp_path
                    / 'raw'
                    / '24F319046907737B_20260417_223541.WAV'
                ),
                'file_name': '24F319046907737B_20260417_223541.WAV',
                'timestamp': datetime(
                    2026, 4, 17, 22, 35, 41, tzinfo=timezone.utc
                ),
                'latitude': 50.432584,
                'longitude': -3.672039,
                'temperature_int_c': 18.5,
                'readable_wav': True,
            }
        ]
    )


def test_birdnet_week_from_timestamp() -> None:
    timestamp = datetime(2026, 4, 17, 22, 35, 41, tzinfo=timezone.utc)

    assert birdnet_week_from_timestamp(timestamp) == 15


def test_load_birdnet_model_uses_python_package(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    loaded_args = {}

    def fake_load(model_kind: str, version: str, backend: str) -> object:
        loaded_args['model_kind'] = model_kind
        loaded_args['version'] = version
        loaded_args['backend'] = backend
        return object()

    fake_birdnet = SimpleNamespace(load=fake_load)
    monkeypatch.setitem(sys.modules, 'birdnet', fake_birdnet)
    monkeypatch.setattr(
        'importlib.util.find_spec',
        lambda name: object() if name == 'birdnet' else None,
    )

    model = load_birdnet_model(config)

    assert model is not None
    assert loaded_args == {
        'model_kind': 'acoustic',
        'version': '2.4',
        'backend': 'tf',
    }


def test_run_birdnet_predictions_filters_by_confidence(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    inventory_df = make_inventory_df(tmp_path)
    model = FakeBirdNETModel()

    predictions_df = run_birdnet_predictions(
        model=model,
        inventory_df=inventory_df,
        config=config,
    )

    assert predictions_df.height == 1
    assert predictions_df['species_name'].to_list() == [
        'Erithacus rubecula_European Robin'
    ]


def test_run_birdnet_predictions_uses_location_species_filter(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    inventory_df = make_inventory_df(tmp_path)
    model = FakeBirdNETModel()
    geo_model = FakeBirdNETGeoModel()

    predictions_df = run_birdnet_predictions(
        model=model,
        inventory_df=inventory_df,
        config=config,
        geo_model=geo_model,
    )

    assert predictions_df.height == 1
    assert geo_model.predict_calls == [(50.432584, -3.672039, 15, 0.03)]
    assert model.predict_kwargs[0]['custom_species_list'] == [
        'Erithacus rubecula_European Robin',
        'Turdus merula_Eurasian Blackbird',
    ]


def test_run_birdnet_predictions_caches_location_species_filter(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    inventory_df = pl.concat(
        [
            make_inventory_df(tmp_path),
            make_inventory_df(tmp_path).with_columns(
                pl.lit(
                    str(tmp_path / 'raw' / '24F319046907737B_20260417_224541.WAV')
                ).alias('file_path'),
                pl.lit('24F319046907737B_20260417_224541.WAV').alias('file_name'),
            ),
        ]
    )
    model = FakeBirdNETModel()
    geo_model = FakeBirdNETGeoModel()

    run_birdnet_predictions(
        model=model,
        inventory_df=inventory_df,
        config=config,
        geo_model=geo_model,
    )

    assert len(geo_model.predict_calls) == 1
    assert len(model.predict_kwargs) == 2


def test_run_birdnet_predictions_writes_location_species_csv(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    inventory_df = make_inventory_df(tmp_path)
    model = FakeBirdNETModel()
    geo_model = FakeBirdNETGeoModel()

    run_birdnet_predictions(
        model=model,
        inventory_df=inventory_df,
        config=config,
        geo_model=geo_model,
        location_species_output_dir=config.birdnet.output_dir,
    )

    species_csv_path = config.birdnet.output_dir / 'birdnet_location_species.csv'
    species_df = pl.read_csv(species_csv_path)

    assert species_csv_path.exists()
    assert species_df.select(
        [
            'latitude',
            'longitude',
            'birdnet_week',
            'species_name',
            'scientific_name',
            'common_name',
            'location_confidence',
        ]
    ).to_dicts() == [
        {
            'latitude': 50.432584,
            'longitude': -3.672039,
            'birdnet_week': 15,
            'species_name': 'Erithacus rubecula_European Robin',
            'scientific_name': 'Erithacus rubecula',
            'common_name': 'European Robin',
            'location_confidence': 0.812,
        },
        {
            'latitude': 50.432584,
            'longitude': -3.672039,
            'birdnet_week': 15,
            'species_name': 'Turdus merula_Eurasian Blackbird',
            'scientific_name': 'Turdus merula',
            'common_name': 'Eurasian Blackbird',
            'location_confidence': 0.746,
        },
    ]


def test_run_birdnet_predictions_post_filters_when_model_rejects_species_list(
    tmp_path: Path,
) -> None:
    class BareOnlyBirdNETModel:
        def __init__(self) -> None:
            self.predict_calls: list[dict[str, object]] = []

        def predict(self, audio_path: str, **kwargs: object) -> pl.DataFrame:
            self.predict_calls.append(kwargs)
            if kwargs:
                raise TypeError('unexpected keyword argument')

            return pl.DataFrame(
                [
                    {
                        'input': audio_path,
                        'start_time': '00:00:03.00',
                        'end_time': '00:00:06.00',
                        'species_name': 'Erithacus rubecula_European Robin',
                        'confidence': 0.876,
                    },
                    {
                        'input': audio_path,
                        'start_time': '00:00:06.00',
                        'end_time': '00:00:09.00',
                        'species_name': 'Cardinalis cardinalis_Northern Cardinal',
                        'confidence': 0.91,
                    },
                ]
            )

    config = make_config(tmp_path)
    inventory_df = make_inventory_df(tmp_path)
    model = BareOnlyBirdNETModel()
    geo_model = FakeBirdNETGeoModel()

    predictions_df = run_birdnet_predictions(
        model=model,
        inventory_df=inventory_df,
        config=config,
        geo_model=geo_model,
    )

    assert predictions_df['species_name'].to_list() == [
        'Erithacus rubecula_European Robin'
    ]
    assert model.predict_calls[-1] == {}


def test_run_birdnet_predictions_handles_numpy_structured_array(
    tmp_path: Path,
) -> None:
    class NumpyBirdNETModel:
        def predict(self, audio_path: str, **kwargs: object) -> np.ndarray:
            return np.array(
                [
                    (
                        audio_path,
                        '00:00:03.00',
                        '00:00:06.00',
                        'Erithacus rubecula_European Robin',
                        0.876,
                    )
                ],
                dtype=[
                    ('input', 'U256'),
                    ('start_time', 'U16'),
                    ('end_time', 'U16'),
                    ('species_name', 'U128'),
                    ('confidence', 'f8'),
                ],
            )

    config = make_config(tmp_path)
    inventory_df = make_inventory_df(tmp_path)

    predictions_df = run_birdnet_predictions(
        model=NumpyBirdNETModel(),
        inventory_df=inventory_df,
        config=config,
    )

    assert predictions_df.height == 1
    assert predictions_df['confidence'].to_list() == [0.876]


def test_run_birdnet_predictions_handles_numpy_structured_rows(
    tmp_path: Path,
) -> None:
    class NumpyRowBirdNETModel:
        def predict(self, audio_path: str, **kwargs: object) -> list[np.void]:
            predictions = np.array(
                [
                    (
                        audio_path,
                        '00:00:03.00',
                        '00:00:06.00',
                        'Erithacus rubecula_European Robin',
                        0.876,
                    )
                ],
                dtype=[
                    ('input', 'U256'),
                    ('start_time', 'U16'),
                    ('end_time', 'U16'),
                    ('species_name', 'U128'),
                    ('confidence', 'f8'),
                ],
            )
            return list(predictions)

    config = make_config(tmp_path)
    inventory_df = make_inventory_df(tmp_path)

    predictions_df = run_birdnet_predictions(
        model=NumpyRowBirdNETModel(),
        inventory_df=inventory_df,
        config=config,
    )

    assert predictions_df.height == 1
    assert predictions_df['confidence'].to_list() == [0.876]


def test_normalise_birdnet_predictions_adds_inventory_metadata_and_temperature(
    tmp_path: Path,
) -> None:
    inventory_df = make_inventory_df(tmp_path)
    audio_path = inventory_df.row(0, named=True)['file_path']
    predictions = pl.DataFrame(
        [
            {
                'input': audio_path,
                'start_time': '00:00:03.00',
                'end_time': '00:00:06.00',
                'species_name': 'Erithacus rubecula_European Robin',
                'confidence': 0.876,
            }
        ]
    )

    detections_df = normalise_birdnet_predictions(
        predictions=predictions,
        inventory_df=inventory_df,
        model_name='acoustic-2.4-tf',
    )

    assert detections_df.height == 1
    detection = detections_df.row(0, named=True)
    assert detection['file_name'] == '24F319046907737B_20260417_223541.WAV'
    assert detection['detection_start_s'] == 3.0
    assert detection['detection_end_s'] == 6.0
    assert detection['scientific_name'] == 'Erithacus rubecula'
    assert detection['common_name'] == 'European Robin'
    assert detection['confidence'] == 0.876
    assert detection['latitude'] == 50.432584
    assert detection['longitude'] == -3.672039
    assert detection['temperature_int_c'] == 18.5
    assert detection['analysis_backend'] == BIRDNET_BACKEND
    assert detection['model_name'] == 'acoustic-2.4-tf'


def test_run_birdnet_analysis_writes_and_reuses_checkpoints(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    inventory_df = make_inventory_df(tmp_path)
    model = FakeBirdNETModel()

    monkeypatch.setattr(
        'audio_ecology.analysis.birdnet.load_birdnet_model',
        lambda config: model,
    )

    detections_df = run_birdnet_analysis(
        config=config,
        inventory_df=inventory_df,
    )

    assert detections_df.height == 1
    assert model.audio_paths == [inventory_df.row(0, named=True)['file_path']]

    checkpoint_store = get_birdnet_checkpoint_store(config)
    audio_path = Path(inventory_df.row(0, named=True)['file_path'])
    assert checkpoint_store.exists(audio_path) is True

    class FailingBirdNETModel:
        def predict(self, audio_path: str, **kwargs: object) -> pl.DataFrame:
            raise AssertionError('Checkpoint should have been reused')

    monkeypatch.setattr(
        'audio_ecology.analysis.birdnet.load_birdnet_model',
        lambda config: FailingBirdNETModel(),
    )

    resumed_df = run_birdnet_analysis(
        config=config,
        inventory_df=inventory_df,
    )

    assert resumed_df.to_dicts() == detections_df.to_dicts()


def test_run_birdnet_analysis_writes_csv_when_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = make_config(tmp_path)
    config.outputs = OutputConfig(write_csv=True)
    inventory_df = make_inventory_df(tmp_path)

    monkeypatch.setattr(
        'audio_ecology.analysis.birdnet.load_birdnet_model',
        lambda config: FakeBirdNETModel(),
    )

    run_birdnet_analysis(
        config=config,
        inventory_df=inventory_df,
    )

    output_dir = get_birdnet_output_dir(config)
    assert (output_dir / 'birdnet_detections.parquet').exists()
    assert (output_dir / 'birdnet_detections.csv').exists()
