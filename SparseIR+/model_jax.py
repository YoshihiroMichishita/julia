import jax, optax
from jax import random, numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints

#if torch.cuda.is_available():
#    tensor = tensor.to("cuda")


# レイヤー定義
class AttentionHead(nn.Module):
    head_dim: int

    def scaled_dot_product_attention(self, q, k, v, mask):
        #scores = jnp.matmul(q, jnp.transpose(k, (0, 2, 1)))
        scores = jnp.matmul(jnp.transpose(k, (0, 2, 1)), q)
        if mask is not None:
            mask = jnp.tile(mask, mask.shape[-1]).reshape(
                    mask.shape[0], -1, mask.shape[-1]) # *7
            scores = jnp.where(mask==0, -jnp.inf, scores) # *8
        #w = nn.softmax(scores / jnp.sqrt(self.head_dim)) # *5
        w = nn.softmax(scores / jnp.sqrt(self.head_dim), axis=-2)
        #return jnp.matmul(w, v) # *6
        return jnp.matmul(v, w)

    @nn.compact
    def __call__(self, hidden_state, attention_mask):
        q = nn.Dense(features=self.head_dim)(hidden_state) # *1
        k = nn.Dense(features=self.head_dim)(hidden_state) # *2
        v = nn.Dense(features=self.head_dim)(hidden_state) # *3
        output = self.scaled_dot_product_attention(q, k, v, attention_mask)
        return output
    
class MultiHeadAttention(nn.Module):
    num_heads: int
    embed_dim: int

    def setup(self):
        head_dim=self.embed_dim // self.num_heads # *1
        self.attention_heads = [AttentionHead(head_dim=head_dim)
                                for _ in jnp.arange(self.num_heads)] # *2

    @nn.compact
    def __call__(self, hidden_state, attention_mask):
        attention_outputs = [head(hidden_state, attention_mask)
                             for head in self.attention_heads] # *3
        x = jnp.concatenate(attention_outputs, axis=-1) # *4
        x = nn.Dense(features=self.embed_dim)(x) # *5
        return x

"""
class MuFilter(nn.Module):
    @nn.compact
    def __call__(self, input):
        x = nn.Dense(features=1)(input)
        x = nn.tanh(x)
        return x

class logsFilter(nn.Module):
    @nn.compact
    def __call__(self, input):
        #x = nn.Dense(features=1)(input)
        x = nn.Dense(features=1)(input)
        x = 4*nn.tanh(x)
        return x
    
class WeightFilter(nn.Module):
    @nn.compact
    def __call__(self, input):
        #x = nn.Dense(features=1)(input)
        x = nn.Dense(features=1)(input)
        x = 4*nn.tanh(x)
        x = nn.softmax(x)
        return x
    
class GaussianFilter(nn.Module):
    def setup(self):
        self.params = [MuFilter(), logsFilter(), WeightFilter()]

    @nn.compact
    def __call__(self, input):
        outputs = [cas(input) for cas in self.Cathods]
        x = jnp.concatenate(outputs, axis=-1)
        return x
"""


class MuFilterS(nn.Module):
    n_gauss: int

    @nn.compact
    def __call__(self, input):
        x = nn.Dense(features=self.n_gauss)(input)
        x = nn.tanh(x)
        return x

class logsFilterS(nn.Module):
    n_gauss: int

    @nn.compact
    def __call__(self, input):
        #x = nn.Dense(features=1)(input)
        x = nn.Dense(features=self.n_gauss)(input)
        x = 4*nn.tanh(x)
        return x
    
class WeightFilterS(nn.Module):
    n_gauss: int

    @nn.compact
    def __call__(self, input):
        #x = nn.Dense(features=1)(input)
        x = nn.Dense(features=self.n_gauss)(input)
        x = 4*nn.tanh(x)
        x = nn.softmax(x)
        return x
        
class OutputFilter(nn.Module):
    n_gauss: int

    def setup(self):
        self.Gaussian = [MuFilterS(n_gauss=self.n_gauss), logsFilterS(n_gauss=self.n_gauss), WeightFilterS(n_gauss=self.n_gauss)]
        #cas_num = 4
        #self.Cathods = [CasFilter() for _ in jnp.arange(cas_num)]

    @nn.compact
    def __call__(self, input):
        outputs = [cas(input) for cas in self.Gaussian]
        x = jnp.concatenate(outputs, axis=-1)
        return x

class FeedForward(nn.Module):
    embed_dim: int
    intermediate_size: int = 128

    @nn.compact
    def __call__(self, x, eval):
        x = nn.Dense(features=self.intermediate_size)(x)
        #x = nn.relu(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.embed_dim)(x)
        #x = nn.Dropout(0.1, deterministic=eval)(x)
        return x
    
class TransformerEncoderBlock(nn.Module):
    num_heads: int
    embed_dim: int

    def setup(self):
        self.attention = MultiHeadAttention(
            num_heads=self.num_heads, embed_dim=self.embed_dim)
        self.feed_forward = FeedForward(embed_dim=self.embed_dim)

    @nn.compact
    def __call__(self, x, attention_mask, eval):
        x = x + self.attention(x, attention_mask) # Skip connection *1
        x = nn.RMSNorm()(x) # *2
        x = x + self.feed_forward(x, eval) # Skip connection *3
        x = nn.RMSNorm()(x) # *4
        return x
    
class TransformerEncoder(nn.Module):
    n_gauss: int
    num_heads: int
    embed_dim: int
    num_hidden_layers: int

    def setup(self):
        #self.embeddings = Embeddings(self.embed_dim)
        self.layers = [TransformerEncoderBlock(num_heads=self.num_heads,
                                               embed_dim=self.embed_dim)
                       for _ in range(self.num_hidden_layers)] # *1
        self.output = OutputFilter(n_gauss=self.n_gauss)

    def __call__(self, input_ids, attention_mask=None, eval=True):
        x = input_ids
        #self.embeddings(input_ids, eval) # *2
        for layer in self.layers: # *3
            x = layer(x, attention_mask, eval)

        x = self.output(x)
        return x

