from typing import Dict, List, Optional
import torch.nn as nn
import torch

ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
    'gelu': nn.GELU(),
    'prelu': nn.PReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh(),
    'softmax': nn.Softmax(dim=-1),
    'gelu': nn.GELU(),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'swish': nn.SiLU(), 
    'identity': nn.Identity()
}

def make_mlp(
    layer_dims: List[int],
    inter_layer_activation_function: str = 'relu',
    final_activation_function: Optional[str] = None
) -> nn.Sequential:

    # Ensure the activation functions are valid
    if inter_layer_activation_function not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Invalid activation function: {inter_layer_activation_function}")
    if final_activation_function is not None and final_activation_function not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Invalid final activation function: {final_activation_function}")
    
    layers = []

    # Iterate through the layer dimensions and create Linear layers
    for i in range(len(layer_dims) - 1):
        # Add a linear layer
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        
        # Add the inter-layer activation function after each hidden layer (except for the last layer)
        if i < len(layer_dims) - 2:
            layers.append(ACTIVATION_FUNCTIONS[inter_layer_activation_function])
    
    # Optionally add the final activation function
    if final_activation_function is not None:
        layers.append(ACTIVATION_FUNCTIONS[final_activation_function])
    
    # Return the Sequential container
    return nn.Sequential(*layers)
    

class MLPModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_layer_dims: List[int],
        output_dim: int,
        inter_layer_activation_function: str,
        final_activation_function: Optional[str] = None,
        input_dropout_rate: float = 0.0,
        input_batch_norm_features: List[int] = [],
    ) -> None:
        super(MLPModel, self).__init__()

        # store init args
        self.input_dim = input_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.output_dim = output_dim
        self.inter_layer_activation_function = inter_layer_activation_function
        self.final_activation_function = final_activation_function
        self.input_dropout_rate = input_dropout_rate
        self.input_batch_norm_features = input_batch_norm_features

        # input dropout layer
        if self.input_dropout_rate:
            self.input_dropout = nn.Dropout(p=self.input_dropout_rate)
        
        # input batch normalization
        if self.input_batch_norm_features:
            self.batch_norm = nn.BatchNorm1d(num_features=len(self.input_batch_norm_features))

        # defining the model
        self.model = make_mlp(
            layer_dims=[self.input_dim] + self.hidden_layer_dims + [self.output_dim],
            inter_layer_activation_function=self.inter_layer_activation_function,
            final_activation_function=self.final_activation_function,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x.clone()

        # Apply batch normalization to the subset of features
        if self.input_batch_norm_features:
            selected_features = x[:, self.input_batch_norm_features]  # Extract subset
            normalized_features = self.batch_norm(selected_features)  # Normalize
            x[:, self.input_batch_norm_features] = normalized_features  # Reinsert

        # Apply dropout to the entire input
        if self.input_dropout_rate:
            x = self.input_dropout(x)

        return self.model(x)


class MLPMeanPoolModel(MLPModel):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x=x)
        return  torch.mean(output, dim=len(x.shape) - 2)

class MultiClassMLPMeanPoolModel(MLPMeanPoolModel):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = super().forward(x=x)
        return  torch.softmax(output, dim=-1)
        


class DeepAttentionModule(nn.Module):

    def __init__(
            self,
            input_dim:int,
            temperature:float = 1.0,
            dropout_rate: float = 0.0,
            xavier_initialization: bool = False
        ):
        super(DeepAttentionModule, self).__init__()
        
        assert temperature > 1e-6, "Temperature must be above 1e-6."

        self.input_dim = input_dim
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.xavier_initialization = xavier_initialization

        # defining layers
        self.theta1 = nn.Linear(in_features=self.input_dim, out_features=self.input_dim)
        self.theta2 = nn.Linear(in_features=self.input_dim, out_features=self.input_dim)

        # NOTE: fix this for multi-class classification
        self.W = nn.Linear(in_features=self.input_dim, out_features=1)

        if self.dropout_rate:
            self.dropout = nn.Dropout(p=self.dropout_rate)

        if self.xavier_initialization:
            self._initialize_weights()


    def forward(self, x):

        A_1 = torch.tanh(self.theta1(x))
        A_2 = torch.sigmoid(self.theta2(x))

        if self.dropout_rate:
            A_1 = self.dropout(A_1)
            A_2 = self.dropout(A_2)

        A = A_1 * A_2
        A = self.W(A)
        A = A / self.temperature
        A = torch.softmax(A, dim=1).squeeze(-1)
        return A
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        def init_weights(layer):
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

        print("Xavier initialization.")

        # Apply initialization to all layers
        self.theta1.apply(init_weights)
        self.theta2.apply(init_weights)
        self.W.apply(init_weights)
    

class MultiClassDeepAttentionModule(nn.Module):

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
        xavier_initialization: bool = False,
    ) -> None:
        super(MultiClassDeepAttentionModule, self).__init__()
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.xavier_initialization = xavier_initialization 

        self.attention_heads = nn.ModuleList()
    
        for _ in range(self.n_classes):
            self.attention_heads.append(
                DeepAttentionModule(
                    input_dim=input_dim,
                    temperature=temperature,
                    dropout_rate=dropout_rate,
                    xavier_initialization=xavier_initialization,
                )
            )

    def forward(self, x):
        attention_scores = []
        for attention_head in self.attention_heads:
            attention_scores.append(attention_head(x))
        return attention_scores


class DeepAttentionMLPModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_layer_dims: List[int],
        output_dim: int,
        inter_layer_activation_function: str,
        final_activation_function: str,
        attention_temp: float=1,
        input_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        xavier_initialization: bool = False,
        input_batch_norm_features: List[int] = []

    ):
        super(DeepAttentionMLPModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.output_dim = output_dim
        self.inter_layer_activation_function = inter_layer_activation_function
        self.final_activation_function = final_activation_function
        self.attention_temp = attention_temp
        self.input_dropout_rate = input_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.xavier_initialization = xavier_initialization
        self.input_batch_norm_features = input_batch_norm_features

        # input batch normalization
        if self.input_batch_norm_features:
            self.batch_norm = nn.BatchNorm1d(num_features=len(self.input_batch_norm_features))
        else:
            self.batch_norm = None

        if self.input_dropout_rate:
            self.input_dropout = nn.Dropout(p=self.input_dropout_rate)

        self.deep_attention = DeepAttentionModule(
            input_dim=self.input_dim,
            temperature=self.attention_temp,
            dropout_rate=self.attention_dropout_rate,
            xavier_initialization=self.xavier_initialization,
        )
        
        self.mlp = make_mlp(
            layer_dims=[self.input_dim] + self.hidden_layer_dims + [self.output_dim],
            inter_layer_activation_function=self.inter_layer_activation_function,
            final_activation_function=self.final_activation_function,
        )

    def forward(self, x):

        x = x.clone()

        # Apply batch normalization to the subset of features
        if self.input_batch_norm_features:
            selected_features = x[:, :, self.input_batch_norm_features]  # Extract subset
            x_permuted = selected_features.permute(0, 2, 1)
            normalized_features = self.batch_norm(x_permuted)  # Normalize
            normalized_features = normalized_features.permute(0, 2, 1)
            x[:, :, self.input_batch_norm_features] = normalized_features  # Reinsert

        if self.input_dropout_rate:
            x = self.input_dropout(x)

        A = self.deep_attention(x)
        bag_x = x * A.unsqueeze(-1)
        bag_x = torch.sum(bag_x, dim=1)
        y_hat = self.mlp(bag_x)
        
        # NOTE: used for model debugging
        tile_level_y_hat = self.mlp(x)

        return y_hat
    

class MultiClassDeepAttentionModule(nn.Module):

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
        xavier_initialization: bool = False,
    ) -> None:
        super(MultiClassDeepAttentionModule, self).__init__()
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.temperature = temperature
        self.dropout_rate = dropout_rate
        self.xavier_initialization = xavier_initialization 

        self.attention_heads = nn.ModuleList()
    
        for _ in range(self.n_classes):
            self.attention_heads.append(
                DeepAttentionModule(
                    input_dim=input_dim,
                    temperature=temperature,
                    dropout_rate=dropout_rate,
                    xavier_initialization=xavier_initialization,
                )
            )

    def forward(self, x):
        attention_scores = []
        for attention_head in self.attention_heads:
            attention_scores.append(attention_head(x))
        return attention_scores


class SpecialDeepAttentionMLPModel(nn.Module):

    ## NOTE: VERY TMP MODEL TO INJECT EHR INTO LATER SECTION OF NETWORK

    def __init__(
        self,
        input_dim: int,
        hidden_layer_dims_1: List[int],
        hidden_layer_dims_2: List[int],
        skip_features:  List[int],
        split_index: int,
        output_dim: int,
        inter_layer_activation_function: str,
        final_activation_function: str,
        attention_temp: float=1,
        input_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        xavier_initialization: bool = False,
        input_batch_norm_features: List[int] = []

    ):
        super(SpecialDeepAttentionMLPModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_layer_dims_1 = hidden_layer_dims_1
        self.hidden_layer_dims_2 = hidden_layer_dims_2
        self.skip_features = skip_features
        self.output_dim = output_dim
        self.inter_layer_activation_function = inter_layer_activation_function
        self.final_activation_function = final_activation_function
        self.attention_temp = attention_temp
        self.input_dropout_rate = input_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.xavier_initialization = xavier_initialization
        self.input_batch_norm_features = input_batch_norm_features
        self.split_index = split_index

        # input batch normalization
        if self.input_batch_norm_features:
            self.batch_norm = nn.BatchNorm1d(num_features=len(self.input_batch_norm_features))
        else:
            self.batch_norm = None

        if self.input_dropout_rate:
            self.input_dropout = nn.Dropout(p=self.input_dropout_rate)

        self.deep_attention = DeepAttentionModule(
            input_dim=self.input_dim,
            temperature=self.attention_temp,
            dropout_rate=self.attention_dropout_rate,
            xavier_initialization=self.xavier_initialization,
        )
        
        self.mlp_1 = make_mlp(
            layer_dims=[split_index] + self.hidden_layer_dims_1,
            inter_layer_activation_function=self.inter_layer_activation_function,
            final_activation_function=self.inter_layer_activation_function,
        )

        self.mlp_2 = make_mlp(
            layer_dims= [self.hidden_layer_dims_1[-1] + len(self.skip_features)] + self.hidden_layer_dims_2 + [self.output_dim],
            inter_layer_activation_function=self.inter_layer_activation_function,
            final_activation_function=self.final_activation_function,
        )


    def forward(self, x):

        x = x.clone()

        # Apply batch normalization to the subset of features
        if self.input_batch_norm_features:
            selected_features = x[:, :, self.input_batch_norm_features]  # Extract subset
            x_permuted = selected_features.permute(0, 2, 1)
            normalized_features = self.batch_norm(x_permuted)  # Normalize
            normalized_features = normalized_features.permute(0, 2, 1)
            x[:, :, self.input_batch_norm_features] = normalized_features  # Reinsert

        if self.input_dropout_rate:
            x = self.input_dropout(x) 

        A = self.deep_attention(x)
        bag_x = x * A.unsqueeze(-1)
        bag_x = torch.sum(bag_x, dim=1)
        
        bag_x_1 = self.mlp_1(bag_x[:, :self.split_index])

        bag_x_1 = torch.concat([bag_x_1, bag_x[:, self.split_index:]], dim=1)

        y_hat = self.mlp_2(bag_x_1)
        
        return y_hat


class MultiClassDeepAttentionMLPModel(nn.Module):

    def __init__(
        self,
        n_classes: int,
        input_dim: int,
        hidden_layer_dims: List[int],
        inter_layer_activation_function: str,
        dropout_rate: float = 0.0,
        attention_temp: float = 1.0,
        xavier_initialization: bool = False,
        input_batch_norm_features: List[int] = []
    ) -> None:
        super(MultiClassDeepAttentionMLPModel, self).__init__()

        assert n_classes >= 2, "Number of classes must be at least 2."

        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_layer_dims = hidden_layer_dims
        self.inter_layer_activation_function = inter_layer_activation_function
        self.attention_temp = attention_temp
        self.dropout_rate = dropout_rate
        self.xavier_initialization = xavier_initialization
        self.input_batch_norm_features = input_batch_norm_features


        # input batch normalization
        if self.input_batch_norm_features:
            self.batch_norm = nn.BatchNorm1d(num_features=len(self.input_batch_norm_features))
        else:
            self.batch_norm = None

        self.multi_class_deep_attention = MultiClassDeepAttentionModule(
            n_classes=self.n_classes,
            input_dim=self.input_dim,
            temperature=self.attention_temp,
            dropout_rate=self.dropout_rate,
            xavier_initialization=self.xavier_initialization,
        )

        self.classifiers = nn.ModuleList()

        for _ in range(self.n_classes):
            self.classifiers.append(
                make_mlp(
                    layer_dims=[self.input_dim] + self.hidden_layer_dims + [1],
                    inter_layer_activation_function=self.inter_layer_activation_function,
                )
            )

    def forward(self, x):

        x = x.clone()

        # Apply batch normalization to the subset of features
        if self.input_batch_norm_features:
            selected_features = x[:, :, self.input_batch_norm_features]  # Extract subset
            x_permuted = selected_features.permute(0, 2, 1)
            normalized_features = self.batch_norm(x_permuted)  # Normalize
            normalized_features = normalized_features.permute(0, 2, 1)
            x[:, :, self.input_batch_norm_features] = normalized_features  # Reinsert

        class_attention_list = self.multi_class_deep_attention(x)
        class_logits_list = []

        for A, classifier in zip(class_attention_list, self.classifiers):
            x_class = x
            x_class = x_class * A.unsqueeze(-1)
            x_class = torch.sum(x_class, dim=1)
            y_hat_class = classifier(x_class)
            class_logits_list.append(y_hat_class)

        y_hat = torch.cat(tuple(class_logits_list), dim=-1)
        y_hat = torch.softmax(y_hat, dim=-1)

        return y_hat
    

class SpecialMultiClassDeepAttentionMLPModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        input_dim: int,
        hidden_layer_dims_1: List[int],
        hidden_layer_dims_2: List[int],
        skip_features: List[int],
        split_index: int,
        inter_layer_activation_function: str,
        attention_temp: float = 1.0,
        input_dropout_rate: float = 0.0,
        attention_dropout_rate: float = 0.0,
        xavier_initialization: bool = False,
        input_batch_norm_features: List[int] = [],
    ) -> None:
        super(SpecialMultiClassDeepAttentionMLPModel, self).__init__()

        assert n_classes >= 2, "Number of classes must be at least 2."

        self.n_classes = n_classes
        self.input_dim = input_dim
        self.hidden_layer_dims_1 = hidden_layer_dims_1
        self.hidden_layer_dims_2 = hidden_layer_dims_2
        self.skip_features = skip_features
        self.split_index = split_index
        self.inter_layer_activation_function = inter_layer_activation_function
        self.attention_temp = attention_temp
        self.input_dropout_rate = input_dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.xavier_initialization = xavier_initialization
        self.input_batch_norm_features = input_batch_norm_features

        # input batch normalization
        if self.input_batch_norm_features:
            self.batch_norm = nn.BatchNorm1d(num_features=len(self.input_batch_norm_features))
        else:
            self.batch_norm = None

        if self.input_dropout_rate:
            self.input_dropout = nn.Dropout(p=self.input_dropout_rate)

        self.multi_class_deep_attention = MultiClassDeepAttentionModule(
            n_classes=self.n_classes,
            input_dim=self.input_dim,
            temperature=self.attention_temp,
            dropout_rate=self.attention_dropout_rate,
            xavier_initialization=self.xavier_initialization,
        )

        self.class_mlp_1_layers = nn.ModuleList()
        self.class_mlp_2_layers = nn.ModuleList()

        for _ in range(n_classes):
            mlp_1 = make_mlp(
                layer_dims=[split_index] + hidden_layer_dims_1,
                inter_layer_activation_function=self.inter_layer_activation_function,
                final_activation_function=self.inter_layer_activation_function,
            )

            mlp_2 = make_mlp(
                layer_dims=[hidden_layer_dims_1[-1] + len(self.skip_features)] + hidden_layer_dims_2 + [1],
                inter_layer_activation_function=self.inter_layer_activation_function,
            )

            self.class_mlp_1_layers.append(mlp_1)
            self.class_mlp_2_layers.append(mlp_2)

    def forward(self, x):
        x = x.clone()

        if self.input_batch_norm_features:
            selected_features = x[:, :, self.input_batch_norm_features]
            x_permuted = selected_features.permute(0, 2, 1)
            normalized_features = self.batch_norm(x_permuted)
            normalized_features = normalized_features.permute(0, 2, 1)
            x[:, :, self.input_batch_norm_features] = normalized_features

        if self.input_dropout_rate:
            x = self.input_dropout(x)

        class_attention_list = self.multi_class_deep_attention(x)
        class_logits_list = []

        for A, mlp_1, mlp_2 in zip(class_attention_list, self.class_mlp_1_layers, self.class_mlp_2_layers):
            bag_x = x * A.unsqueeze(-1)
            bag_x = torch.sum(bag_x, dim=1)

            bag_x_1 = mlp_1(bag_x[:, :self.split_index])
            bag_x_1 = torch.cat([bag_x_1, bag_x[:, self.split_index:]], dim=1)
            y_hat_class = mlp_2(bag_x_1)

            class_logits_list.append(y_hat_class)

        y_hat = torch.cat(class_logits_list, dim=-1)
        y_hat = torch.softmax(y_hat, dim=-1)

        return y_hat



class MultiHeadedDeepAttentionMLPModel(nn.Module):

    def __init__(
        self,
        output_dim: int,
        input_dim: int,
        n_heads: int,
        hidden_layer_dims: List[int],
        inter_layer_activation_function: str,
        final_activation_function: str,
        dropout_rate: float = 0.0,
        attention_temp: float = 1.0,
        xavier_initialization: bool = False,
    ) -> None:
        super(MultiHeadedDeepAttentionMLPModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.hidden_layer_dims = hidden_layer_dims
        self.inter_layer_activation_function = inter_layer_activation_function
        self.final_activation_function = final_activation_function
        self.attention_temp = attention_temp
        self.dropout_rate = dropout_rate
        self.xavier_initialization = xavier_initialization

        self.multi_headed_deep_attention = MultiClassDeepAttentionModule(
            n_classes=self.n_heads,
            input_dim=self.input_dim,
            temperature=self.attention_temp,
            dropout_rate=self.dropout_rate,
            xavier_initialization=self.xavier_initialization,
        )

        self.predictor = make_mlp(
            layer_dims=[self.input_dim * self.n_heads] + self.hidden_layer_dims + [self.output_dim],
            inter_layer_activation_function=self.inter_layer_activation_function,
            final_activation_function=self.final_activation_function,
        )

    def forward(self, x):

        attention_list = self.multi_headed_deep_attention(x)
        x_cat = []
        for A in attention_list:
            x_cat.append(torch.sum(x * A.unsqueeze(-1), dim=1))
        x_cat = torch.cat(tensors=x_cat, dim=1)
        y_hat = self.predictor(x_cat)
        return y_hat


class MultiModelWrapper(nn.Module):

    def __init__(
        self,
        base_model_type: str,
        n_models: int,
        input_data_dim_list: List[int],
        default_model_init_args: Dict,
        unique_model_init_args: Dict = {},
    ) -> None:
        super(MultiModelWrapper, self).__init__()

        # sanity checks
        assert n_models > 1, "Number of base models must be more than 1"
        assert len(input_data_dim_list) == n_models, "len(input_data_dim_list) must match n_models"

        # saving arguments
        self.n_models = n_models
        self.base_model_type = base_model_type
        self.default_model_init_args = default_model_init_args
        self.unique_model_init_args = unique_model_init_args
        self.input_data_dim_list = input_data_dim_list

        # constructing the list of models
        self.models = []
        for model_idx in range(self.n_models):
            unique_model_args = {k: v[model_idx] for k, v in self.unique_model_init_args.items()}
            model = setup_model(
                model_type=self.base_model_type, 
                model_init_args=self.default_model_init_args | unique_model_args
            )
            self.models.append(model)
        self.models = nn.ModuleList(self.models)

    def forward(self, x):
        y_hat_list = []
        input_idx = 0
        for model_idx, model in enumerate(self.models):
            y_hat_list.append(model(x[..., input_idx: input_idx + self.input_data_dim_list[model_idx]]))
            input_idx += self.input_data_dim_list[model_idx]

        # averaging the predictioins across all models
        y_hat_stack = torch.stack(y_hat_list, dim=0)
        y_hat_mean = torch.mean(y_hat_stack, dim=0)

        return y_hat_mean
        
    
# _______________________________________  Model Setup _____________________________________________

MODEL_MAP = {
    "MLPModel": MLPModel,
    "MLPMeanPoolModel": MLPMeanPoolModel,
    "MultiClassMLPMeanPoolModel": MultiClassMLPMeanPoolModel,
    "DeepAttentionMLPModel": DeepAttentionMLPModel,
    "MultiClassDeepAttentionMLPModel": MultiClassDeepAttentionMLPModel,
    "MultiHeadedDeepAttentionMLPModel": MultiHeadedDeepAttentionMLPModel,
    "MultiModelWrapper": MultiModelWrapper,

    #TODO: REMOVE WHEN NOT USEFUL
    "SpecialDeepAttentionMLPModel": SpecialDeepAttentionMLPModel,
    "SpecialMultiClassDeepAttentionMLPModel": SpecialMultiClassDeepAttentionMLPModel,
}

def setup_model(model_type: str, model_init_args: Dict) -> nn.Module:

    if model_type not in MODEL_MAP:
        raise ValueError(f"{model_type} model not founds. Select from {list(MODEL_MAP.keys())}")
    
    return MODEL_MAP[model_type](**model_init_args)
