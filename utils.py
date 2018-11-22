import time
import uuid
import numpy as np
from nltk.tokenize import word_tokenize
import re


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, max_sent_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        max_sent_length:
    Returns:
        a list of list where each sublist has same length
    """
    max_length = max(map(lambda x: len(x), sequences))
    max_length = max_length if max_length < max_sent_length else max_sent_length
    sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)

    return sequence_padded, sequence_length


def parse_raw_data(s):
    s = re.sub(r'[^\w.,?!\'; ]', '', s)
    s = re.sub(r'([.,?!;])(\w)', r'\1 \2', s)
    s = re.sub(r'\s+', ' ', s)
    return word_tokenize(s)


def get_trimmed_w2v_vectors(filename):
    """
    Args:
        filename: path to the npz file
    Returns:
        matrix of embeddings (np array)
    """
    with np.load(filename) as data:
        return data["embeddings"]


def load_vocab(filename):
    """
    Args:
        filename: file with a word per line
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx + 1  # preserve idx 0 for pad_tok

    return d


class Timer:
    def __init__(self):
        self.job_info = {}
        self.running_jobs = []

    def start(self, job):
        if job is None:
            return None

        job_id = str(uuid.uuid4())

        self.job_info[job_id] = (job, time.time())
        print("[INFO] {job} started.".format(job=job))
        self.running_jobs.append(job_id)

        return job_id

    def stop(self, job_id=None):
        if job_id is None and len(self.running_jobs) != 0:
            job_id = self.running_jobs[-1]

        if job_id is None or job_id not in self.job_info:
            return None

        name, start = self.job_info.pop(job_id)
        elapsed_time = time.time() - start
        print("[INFO] {job} finished in {elapsed_time:0.3f}s.".format(job=name, elapsed_time=elapsed_time))
        self.running_jobs.remove(job_id)


class Log:
    verbose = True

    @staticmethod
    def log(text):
        if Log.verbose:
            print(text)
