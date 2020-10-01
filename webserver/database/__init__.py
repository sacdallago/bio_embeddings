import gridfs
from pymongo import MongoClient
from webserver.utilities.configuration import configuration

_client = MongoClient(configuration['web']['mongo_url'])
_collection = _client.file_storage
_fs = gridfs.GridFS(_collection)
_collection.fs.files.create_index('uploadDate', expireAfterSeconds=10*24*60*60)


def write_file(job_id, file_identifier, file_path):
    with open(file_path, "rb") as f:
        index = _fs.put(f, filename=file_identifier, job=job_id)

    return index


def get_file(job_id, file_identifier):
    db_file = _fs.find_one(filter={
        "filename": file_identifier,
        "job": job_id
    })

    if db_file:
        return _fs.get(db_file._id)
    else:
        return None
