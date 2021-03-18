import mongomock
import mongomock.gridfs
import pymongo
import pytest
from flask import url_for
from mongomock.database import Database
from pymongo.errors import CollectionInvalid


def create_collection(self, name, **kwargs):
    """mongomock errors when kwargs are given, but we know it's safe to ignore them"""
    self._ensure_valid_collection_name(name)
    if name in self.list_collection_names():
        raise CollectionInvalid("collection %s already exists" % name)

    self._store.create_collection(name)
    return self[name]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("MODEL_DIRECTORY", "")
    # Create an in-memory broker so we don't need a running celery
    monkeypatch.setenv("CELERY_BROKER_URL", "memory://")
    # Those will be used in webserver.database
    mongomock.gridfs.enable_gridfs_integration()
    monkeypatch.setattr(pymongo, "MongoClient", mongomock.MongoClient)
    # mongomock doesn't support size and capped in create_collection
    monkeypatch.setattr(Database, "create_collection", create_collection)
    from webserver.backend import create_app

    app = create_app()
    return app


def test_app(client, celery_app, celery_worker):
    """Simple 'Does this even start?' test"""
    response = client.get(url_for("index"))

    # TODO: Make this work
    assert "Pipeline jobs are currently inactive." in response.data.decode()
    assert response.status_code == 200
