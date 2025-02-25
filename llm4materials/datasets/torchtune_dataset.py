# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data._utils import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer

from pathlib import Path
from glob import glob 
from tqdm import tqdm
import ase.io
import pandas as pd
from llm4materials.encoders.cartesian import Cartesian

class TextCompletionDataset(Dataset):
    """
    Freeform dataset for any unstructured text corpus. Quickly load any dataset
    from Hugging Face or local disk and tokenize it for your model.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        column (str): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data. For local datasets with a single column
            (e.g. unstructured txt files), use the default "text" which is used by Hugging Face datasets
            when loaded into memory. Default is "text".
        add_eos (bool): Whether to add an EOS token to the end of the sequence. Default is True.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``,
            such as ``data_files`` or ``split``.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        conditional: bool = False,
        add_eos: bool = True,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self.add_eos = add_eos
        self.conditional = conditional

        # load data
        self._data = self.load_data(source)

        # instantiate encoder
        # encoder encodes Atoms object to text representation
        self.encoder = Cartesian()

    def load_data(self, source):
        # read the data source and return a list of ase Atoms object
        cif_files = glob(source + '/*.cif')
        assert len(cif_files) > 0, f"No CIF files found in {source}"
        print(f"Found {len(cif_files)} CIF files in {source}")

        _data = []

        print("Loading CIF files...")
        for cif_file in tqdm(cif_files):
            _data.append({
                "atoms": ase.io.read(cif_file),
                "structure_id": Path(cif_file).stem
            })

        if not self.conditional:
            return _data
        
        # if conditional we need to load the properties as well
        # load the properties
        df = pd.read_csv("targets.csv", header=None, names=["structure_id", "eform", "bandgap", "ehull"])
        eform_dict = dict(zip(df["structure_id"], df["eform"]))
        bandgap_dict = dict(zip(df["structure_id"], df["bandgap"]))

        for each in _data:
            each["eform"] = eform_dict[each["structure_id"]]
            each["bandgap"] = bandgap_dict[each["structure_id"]]
        
        return _data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        # get prompt header for our text representation
        if self.conditional:
            prompt_header = self.encoder.conditional_prompt_header
            prompt_header = prompt_header.replace("<eform>", f"{sample['eform']:.2f}")
            prompt_header = prompt_header.replace("<bandgap>", f"{sample['bandgap']:.2f}")
        else:
            prompt_header = self.encoder.prompt_header

        text_representation = self.encoder.encode(sample['atoms'])

        full_prompt = prompt_header + text_representation

        tokens = self._tokenizer.encode(
            text=full_prompt, 
            add_bos=True, 
            add_eos=self.add_eos
        )

        # Truncate if needed, but don't coerce EOS id
        if self._tokenizer.max_seq_len is not None:
            tokens = truncate(tokens, self._tokenizer.max_seq_len - 1)

        # No need to offset labels by 1 - happens in the recipe
        labels = tokens.copy()

        return {"tokens": tokens, "labels": labels}


def text_completion_dataset(
    tokenizer: ModelTokenizer,
    source: str,
    conditional: bool = False,
    add_eos: bool = True,
    packed: bool = False,
    split_across_pack: bool = True,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[TextCompletionDataset, PackedDataset]:
    """
    Build a configurable dataset from a freeform, unstructured text corpus similar
    to datasets used in pre-training. This method should be
    used to configure a custom text dataset from the yaml config instead of
    using :class:`~torchtune.datasets.TextCompletionDataset` directly, as it is made to be config friendly.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        column (str): name of column in the sample that contains the text data. This is typically required
            for Hugging Face datasets or tabular data. For local datasets with a single column
            (e.g. unstructured txt files), use the default "text" which is used by Hugging Face datasets
            when loaded into memory. Default is "text".
        add_eos (bool): Whether to add an EOS token to the end of the sequence. Default is True.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
        split_across_pack (bool): if the last sample in a pack does not fit in ``max_seq_len``,
            split the sample into the next pack, or move it entirely to the beginning of the next pack.
            For pre-training, typically this is set to True for general text completion. For
            fine-tuning, typically this is set to False to avoid truncating sentences in instruct
            tuning. This argument is ignored if ``packed=False``. Default is True.
        split (str): ``split`` argument for ``datasets.load_dataset``. You can use this argument to load a subset
            of a given split, e.g. ``split="train[:10%]"``. Default is "train".
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Examples:
        >>> from torchtune.datasets import text_completion_dataset
        >>> dataset = text_completion_dataset(
        ...   tokenizer=tokenizer,
        ...   source="allenai/c4",
        ...   column="text",
        ...   data_dir="realnewslike",
        ...   packed=False,
        ...   split="train",
        ... )

    This can also be accomplished via the yaml config::

        dataset:
            _component_: torchtune.datasets.text_completion_dataset
            source: allenai/c4
            column: text
            data_dir: realnewslike
            packed: False
            split: train

    Returns:
        Union[TextCompletionDataset, PackedDataset]: the configured :class:`~torchtune.datasets.TextCompletionDataset`
            or :class:`~torchtune.datasets.PackedDataset` if ``packed=True``

    Raises:
        ValueError: If ``packed=True`` and ``tokenizer.max_seq_len`` is not set.
    """
    ds = TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        conditional=conditional,
        add_eos=add_eos,
        split=split,
        **load_dataset_kwargs,
    )
    if packed:
        if tokenizer.max_seq_len is None:
            raise ValueError(
                "PackedDataset requires a max_seq_len to be set on the tokenizer."
            )
        return PackedDataset(
            ds, max_seq_len=tokenizer.max_seq_len, split_across_pack=split_across_pack
        )
    return ds


if __name__ == "__main__":
    # test the dataset
    # example here uses the llama3 tokenizer
    # check your config fiile for the tokenizer used
    from torchtune.models.llama3 import llama3_tokenizer

    dataset_path = "../../data/mp_20/raw_val"
    tokenizer_path = "/storage/ice1/5/5/sjia65/Meta-Llama-3.1-8B-Instruct/original/tokenizer.model"
    tokenizer = llama3_tokenizer(tokenizer_path)

    ds = text_completion_dataset(
        tokenizer=tokenizer,
        source=dataset_path,
        packed=False,
        split="train",
        conditional=True
    )

    print(ds[0])