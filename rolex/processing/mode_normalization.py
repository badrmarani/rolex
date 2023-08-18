from typing import List, Optional, Union

import pandas as pd
import torch
from ctgan.data_transformer import DataTransformer
from tqdm import trange

from .utils import get_categorical_columns, highly_correlated_features, oversample


class ModeNormalization:
    """
    A class for performing mode-based normalization and transformation on data.

    Args:
            correlation_threshold (float): The threshold for feature correlation removal.
            oversample_quantile (float): The quantile for oversampling.
            qq_threshold (float): The threshold for calculating highly correlated features.
            categorical_columns (Optional[List[str]], optional): List of categorical column names. Defaults to None.
            n_samples_to_transform (Optional[int], optional): Number of samples for data transformation. Defaults to None.
            mode (Optional[str], optional): Mode for transformation ('janus' or None). Defaults to None.
    """

    def __init__(
        self,
        correlation_threshold: float,
        oversample_quantile: float,
        qq_threshold: float,
        categorical_columns: Optional[List[str]] = None,
        n_samples_to_transform: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> None:
        """
        Remove correlated features from the input data.

        Args:
                data (pd.DataFrame): Input data.

        Returns:
                pd.DataFrame: Data with correlated features removed.
        """
        self.correlation_threshold = correlation_threshold
        self.oversample_quantile = oversample_quantile
        self.qq_threshold = qq_threshold
        self.categorical_columns = categorical_columns
        self.n_samples_to_transform = n_samples_to_transform
        self.mode = mode

    def remove_correlated_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data transformation using a DataTransformer.

        Args:
                data (pd.DataFrame): Input data.
                transformer (Optional[DataTransformer], optional): DataTransformer instance. Defaults to None.

        Returns:
                Union[torch.Tensor, DataTransformer]: Transformed data or DataTransformer instance.
        """
        list_correlated_features = highly_correlated_features(
            data=data,
            features=data.columns,
            correlation_threshold=self.correlation_threshold,
        )
        if len(list_correlated_features):
            data = data.loc[:, list_correlated_features]
        return data

    def data_transformer(
        self,
        data: pd.DataFrame,
        transformer: Optional[DataTransformer] = None,
    ) -> Union[torch.Tensor, DataTransformer]:
        """
        Apply data transformation using a DataTransformer.

        Args:
                data (pd.DataFrame): Input data.
                transformer (Optional[DataTransformer], optional): DataTransformer instance. Defaults to None.

        Returns:
                Union[torch.Tensor, DataTransformer]: Transformed data or DataTransformer instance.
        """
        temp_data = data.copy()
        if transformer is None:
            sample_size = (
                min(temp_data.shape[0], 500_000)
                if self.n_samples_to_transform is None
                else self.n_samples_to_transform
            )
            print(
                f"Fitting the `DataTransformer` to a sample of the training data (sample_size={sample_size})"
            )
            transformer = DataTransformer()
            transformer.fit(
                temp_data.sample(sample_size), self.categorical_columns
            )
            return transformer
        else:
            steps = temp_data.shape[0] // 10
            for i in trange(0, temp_data.shape[0], steps, ascii=True):
                xx = temp_data.iloc[i : i + steps, :]
                xx = transformer.transform(xx)
                xx = xx.astype("float64")
                xx = torch.from_numpy(xx)
                if not i:
                    out = xx
                else:
                    out = torch.concatenate((out, xx), dim=0)
            return out

    def fit_transform(self, data: pd.DataFrame) -> torch.Tensor:
        """
        Fit and transform input data.

        Args:
                data (pd.DataFrame): Input data.

        Returns:
                torch.Tensor: Transformed data as a PyTorch tensor.
        """
        x_new = data.copy()
        x_new = self.remove_correlated_features(data)

        if self.mode is not None and self.mode.lower() == "janus":
            x_new = data.loc[
                :,
                list(
                    set(x_new.columns.tolist())
                    | set(
                        [
                            "data_034",
                            "data_037",
                            "data_040",
                            "data_043",
                            "data_046",
                            "data_049",
                        ]
                    )
                ),
            ].astype("float64")
        else:
            x_new = data.astype("float64")

        x_new = oversample(
            x_new, self.categorical_columns, self.oversample_quantile
        )
        if self.categorical_columns is None:
            self.categorical_columns = get_categorical_columns(
                x_new, self.qq_threshold
            )
        elif isinstance(self.categorical_columns, (list, set)):
            raise ValueError
        self.transformer = self.data_transformer(data=x_new, transformer=None)
        x_new = self.data_transformer(data=x_new, transformer=self.transformer)
        return x_new

    def transform(
        self, data: pd.DataFrame, apply_oversampling: bool = True
    ) -> torch.Tensor:
        """
        Transform input data.

        Args:
                data (pd.DataFrame): Input data.
                apply_oversampling (bool, optional): Whether to apply oversampling. Defaults to True.

        Returns:
                torch.Tensor: Transformed data as a PyTorch tensor.
        """
        x_new = data.copy()
        x_new = x_new.loc[
            :,
            [
                c.column_name
                for c in self.transformer._column_transform_info_list
            ],
        ]
        if apply_oversampling:
            x_new = oversample(x_new)
        x_new = self.data_transformer(x_new, transformer=self.transformer)
        return x_new
