"""Separate synthetic backend. Does not read competition materials or real registries."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import UUID

from forestcheck.api.contracts import AnalysisProfile
from forestcheck.api.service import Service
from forestcheck.api.settings import Settings
from forestcheck.ingest.mapping import ImportMapping
from forestcheck.repair.models import PrecisionPolicy
from forestcheck.rules.models import RuleDefinition, RuleSource
from forestcheck.storage.application import Store, digest


def fixture():
    labels = (
        "Синтетический номер выдела",
        "Запас на гектар, м³/га",
        "Площадь синтетического объекта, га",
        "Общий запас древесины исследуемого синтетического объекта с длинным русским заголовком, м³",
        "Примечание",
    )
    mapping = ImportMapping(
        schema_version="1.0",
        mapping_version="ui-synthetic-1",
        source_format="csv",
        encoding="utf-8",
        delimiter=";",
        table="synthetic_ui",
        record_level="stand",
        grain_basis="Only independently created synthetic objects, not forestry methodology",
        columns=tuple(
            dict(
                position=i + 1,
                source_header=labels[i],
                field=name,
                unit=unit,
                transform=transform,
                key_role="stand" if i == 0 else None,
                basis="Synthetic fixture declaration",
            )
            for i, (name, unit, transform) in enumerate(
                (
                    ("stand", None, "text"),
                    ("per", "m3/ha", "decimal_point"),
                    ("area", "ha", "decimal_point"),
                    ("total", "m3", "decimal_point"),
                    ("note", None, "text"),
                )
            )
        ),
    )
    source = RuleSource(
        reference_id="synthetic-ui-fixture",
        locator="Explicit synthetic arithmetic",
        kind="synthetic",
    )
    required = RuleDefinition(
        schema_version="1.0",
        id="synthetic-required",
        version="1",
        record_levels=("stand",),
        categories=("synthetic_forest",),
        dependencies=("total",),
        source=source,
        severity="warning",
        priority=1,
        explanation="Synthetic category requires a total",
        check=dict(kind="required", field="total"),
    )
    balance = RuleDefinition(
        schema_version="1.0",
        id="synthetic-balance",
        version="1",
        record_levels=("stand",),
        dependencies=("per", "area", "total"),
        source=source,
        severity="error",
        priority=1,
        explanation="Declared compatible synthetic quantities",
        check=dict(
            kind="stock_balance",
            per_ha="per",
            area="area",
            total="total",
            quantity_levels=("stand", "stand", "stand"),
            stock_kinds=("live", "live"),
            relationship_confirmed=True,
            tolerance_kind="explicit",
            absolute_tolerance="0",
        ),
    )
    profile = AnalysisProfile(
        schema_version="1.0",
        id=UUID("b588c9aa-f497-4f24-8c7e-a10a3f8ec695"),
        name="Синтетическая арифметика",
        version="1",
        mapping_sha256=digest(mapping.model_dump(mode="json")),
        record_level="stand",
        category="synthetic_forest",
        source_kind="synthetic",
        rules=(required, balance),
        numeric_fields={"total": "m3"},
        precision=(PrecisionPolicy(field_name="total", unit="m3", step="1", source=source),),
        limitations=("synthetic_only",),
    )
    rows = [";".join(labels)]
    for i in range(1, 361):
        rows.append(f"{i:04};10;2;{'' if i % 3 == 1 else '20'};Синтетическая строка {i}")
    return mapping, profile, "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--write-fixtures", action="store_true")
    args = parser.parse_args()
    worker = None
    try:
        mapping, profile, csv = fixture()
        root = Path(__file__).resolve().parents[1]
        if args.write_fixtures:
            target = root / "frontend/public/demo"
            target.mkdir(parents=True, exist_ok=True)
            (target / "synthetic.csv").write_text(csv, "utf-8")
            (target / "mapping.json").write_text(
                json.dumps(mapping.model_dump(mode="json"), ensure_ascii=False, indent=2), "utf-8"
            )
        else:
            os.environ["FORESTCHECK_DATA_MODE"] = "synthetic"
            os.environ["FORESTCHECK_ALLOWED_ORIGINS"] = (
                "http://127.0.0.1:5173,http://127.0.0.1:4173"
            )
            settings = Settings.from_env()
            if settings.root.exists() and any(settings.root.iterdir()):
                raise ValueError("fresh_demo_storage_required")
            store = Store(settings)
            store.initialize()
            # Prevent accidental relabelling of any existing real backend as synthetic.
            with store.transaction() as cx:
                if cx.execute("SELECT count(*) FROM datasets").fetchone()[0]:
                    raise ValueError("fresh_demo_storage_required")
            Service(store).register_profile(profile)
            worker = subprocess.Popen(
                [sys.executable, "-B", str(root / "scripts/backend_local.py"), "worker"],
                cwd=root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            import uvicorn
            from forestcheck.api.app import create_app

            uvicorn.run(
                create_app(settings),
                host="127.0.0.1",
                port=args.port,
                access_log=False,
                log_config=None,
            )
        print("DEMO_COMPLETE")
        return 0
    except Exception:
        print("DEMO_FAILED")
        return 3
    finally:
        if worker is not None:
            worker.terminate()
            worker.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
