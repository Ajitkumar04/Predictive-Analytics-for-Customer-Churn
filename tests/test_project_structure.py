from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_foundation_files_exist():
    assert (ROOT / ".gitignore").exists()
    assert (ROOT / ".env.example").exists()
    assert (ROOT / "configs" / "config.yaml").exists()
    assert (ROOT / "requirements-dev.txt").exists()


def test_core_package_modules_exist():
    assert (ROOT / "src" / "__init__.py").exists()
    assert (ROOT / "src" / "utils" / "__init__.py").exists()
    assert (ROOT / "src" / "utils" / "config.py").exists()
    assert (ROOT / "src" / "utils" / "logger.py").exists()
    assert (ROOT / "src" / "utils" / "exceptions.py").exists()


def test_api_frontend_entries_exist():
    assert (ROOT / "api" / "__init__.py").exists()
    assert (ROOT / "api" / "main.py").exists()
    assert (ROOT / "api" / "schemas.py").exists()
    assert (ROOT / "api" / "routes" / "__init__.py").exists()
    assert (ROOT / "api" / "routes" / "prediction.py").exists()
    assert (ROOT / "frontend" / "app.py").exists()
