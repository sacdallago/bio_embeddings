"""
Abstract interface for Embedder.

Authors:
  Christian Dallago
"""

import abc


class EmbedderInterface(object, metaclass=abc.ABCMeta):

    def __init__(self, model):
        """
        Initializer accepts location of a pre-trained model
        :param model: location of the model
        """
        self.model = model

        pass

    @abc.abstractmethod
    def update_job_status(self, status=None, stage=None):
        """
        Updates the status and/or stage of the job. Status should be EStatus

        Parameters
        ----------
        status EStatus status. Default is None, which won't update it.
        stage string representing compute stage (e.g.: align, compare,...). Default None, which won't update the field

        Returns
        -------
        dictionary representing current job status

        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_jobs_from_group(group_id, connection_string):
        """
        Get all compute jobs from job group.


        Parameters
        ----------
        group_id the group id to look for
        connection_string the connection string for the database of choice

        Returns
        -------
        An array of dictionaries containing fields like name, group_id, etc.

        """
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_job(job_id, connection_string):
        """

        Parameters
        ----------
        job_id the id of the job to look for
        connection_string the connection string for the database of choice

        Returns
        -------
        A dictionary with compute job status
        """
        raise NotImplementedError

    # Properties
    @property
    @abc.abstractmethod
    def job_name(self):
        raise NotImplementedError

    @job_name.getter
    @abc.abstractmethod
    def job_name(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def job_group(self):
        raise NotImplementedError

    @job_group.getter
    @abc.abstractmethod
    def job_group(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def status(self):
        raise NotImplementedError

    @status.getter
    @abc.abstractmethod
    def status(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def stage(self):
        raise NotImplementedError

    @stage.getter
    @abc.abstractmethod
    def stage(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def created_at(self):
        raise NotImplementedError

    @created_at.getter
    @abc.abstractmethod
    def created_at(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def updated_at(self):
        raise NotImplementedError

    @updated_at.getter
    @abc.abstractmethod
    def updated_at(self):
        raise NotImplementedError


class DocumentNotFound(Exception):
    """
    Exception for not finding a document that should be there in the database
    """


DATABASE_NAME = "metadata"