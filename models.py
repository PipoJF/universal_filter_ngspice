import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.stateless as stateless

class AnalogUtils:
    """
    Simula imperfecciones de hardware analógico:
    1. Cuantización (Niveles discretos de conductancia)
    2. Ruido de Peso (Variabilidad térmica/fabricación)
    3. Clipping (Límites de voltaje/conductancia)
    """

    @staticmethod
    def fake_quantize(weights, levels=16, range_limit=1.0):
        """
        Simula la baja resolución de las resistencias programables.
        levels: Cantidad de estados posibles de la resistencia (ej. 16, 32, 64).
        """
        # 1. Clamping: Restringir valores al rango físico [-range, range]
        w_clamped = torch.clamp(weights, -range_limit, range_limit)

        # 2. Escalar al rango de enteros [0, levels-1]
        scale = (levels - 1) / (2 * range_limit)
        w_scaled = (w_clamped + range_limit) * scale

        # 3. Redondear (Simular la discretización) - Usamos .detach() para el round
        # pero mantenemos el gradiente fluyendo (Straight Through Estimator)
        w_rounded = (w_scaled.round() - w_scaled).detach() + w_scaled

        # 4. Des-escalar de vuelta al rango original
        w_quant = (w_rounded / scale) - range_limit
        return w_quant

    @staticmethod
    def inject_noise(weights, std_dev=0.02):
        """
        Agrega ruido gaussiano a los pesos para simular deriva térmica y ruido de lectura.
        std_dev: 0.02 significa 2% de ruido respecto a la escala unitaria.
        """
        noise = torch.randn_like(weights) * std_dev
        return weights + noise

class AnalogLinear(nn.Linear):
    """
    Una capa Linear (Densa) que se comporta como un Crossbar Array analógico.
    Reemplaza nn.Linear con esto.
    """
    def __init__(self, in_features, out_features, bias=True,
                 analog_levels=32, noise_std=0.02):
        super(AnalogLinear, self).__init__(in_features, out_features, bias)
        self.analog_levels = analog_levels
        self.noise_std = noise_std
        self.training_mode = True # Flag para activar/desactivar efectos

    def forward(self, input):
        # 1. Copiamos los pesos originales
        w_simulated = self.weight

        # 2. Aplicamos Cuantización (Si estamos entrenando o validando en modo hardware)
        w_simulated = AnalogUtils.fake_quantize(w_simulated, levels=self.analog_levels)

        # 3. Aplicamos Ruido (Solo si training_mode es True)
        if self.training and self.noise_std > 0:
            w_simulated = AnalogUtils.inject_noise(w_simulated, std_dev=self.noise_std)

        # 4. Operación Lineal usando los pesos "sucios"
        # F.linear usa (input, weight, bias)
        return F.linear(input, w_simulated, self.bias)

class AnalogConv1d(nn.Conv1d):
    """
    Versión analógica de Conv1d.
    Simula que los filtros de convolución están almacenados en memristores/resistencias.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', device=None, dtype=None,
                 analog_levels=32, noise_std=0.02):

        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode, device, dtype)

        self.analog_levels = analog_levels
        self.noise_std = noise_std

    def forward(self, input):
        # 1. Copiamos pesos
        w_simulated = self.weight

        # 2. Cuantización (Simular resolución finita)
        w_simulated = AnalogUtils.fake_quantize(w_simulated, levels=self.analog_levels)

        # 3. Inyección de Ruido (Solo en training)
        if self.training and self.noise_std > 0:
            w_simulated = AnalogUtils.inject_noise(w_simulated, std_dev=self.noise_std)

        # 4. Convolución usando F.conv1d con los pesos sucios
        return nn.functional.conv1d(input, w_simulated, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
    
class AnalogLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 analog_levels=32, noise_std=0.02):
        super(AnalogLSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.noise_std = noise_std

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # AnalogLinear ya es seguro, no necesita cambios
        self.fc = AnalogLinear(hidden_size, output_size,
                               analog_levels=analog_levels,
                               noise_std=noise_std)

    def forward(self, x):
        # Inicializar estados ocultos
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # --- LÓGICA STATELESS (Sin tocar los pesos originales) ---
        if self.training and self.noise_std > 0:
            # 1. Creamos copias temporales ruidosas (Out-of-place)
            noisy_params = {}
            for name, param in self.lstm.named_parameters():
                noise = torch.randn_like(param) * self.noise_std
                noisy_params[name] = param + noise

            # 2. Ejecutamos la LSTM usando los pesos ruidosos explícitos
            # Esto calcula el output y permite que el gradiente fluya a través del ruido
            # sin corromper el parámetro original 'param'.
            out, _ = stateless.functional_call(self.lstm, noisy_params, (x, (h0, c0)))
        else:
            # Ejecución limpia (Validación / Test sin ruido térmico)
            out, _ = self.lstm(x, (h0, c0))

        # --- SALIDA MANY-TO-MANY ---
        # Ya NO hacemos out[:, -1, :]. Pasamos la secuencia completa (batch, 960, hidden)
        out = self.fc(out)

        return out

class AnalogFiLM(nn.Module):
    def __init__(self, channels, knob_dim, analog_levels=32, noise_std=0.02):
        super().__init__()
        # Usamos AnalogLinear para generar los coeficientes
        self.gen = AnalogLinear(knob_dim, channels * 2,
                                analog_levels=analog_levels, noise_std=noise_std)

    def forward(self, x, knobs):
        params = self.gen(knobs).unsqueeze(2)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return x * gamma + beta

class AnalogTemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, knob_dim,
                 analog_levels=32, noise_std=0.02):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        # CONVOLUCIÓN ANALÓGICA
        self.conv = AnalogConv1d(in_ch, out_ch, kernel_size,
                                 padding=self.padding, dilation=dilation,
                                 analog_levels=analog_levels, noise_std=noise_std)

        # FiLM ANALÓGICO
        self.film = AnalogFiLM(out_ch, knob_dim,
                               analog_levels=analog_levels, noise_std=noise_std)

        self.act = nn.PReLU()
        self.norm = nn.GroupNorm(1, out_ch)

        # RESIDUAL ANALÓGICA (Si es necesaria proyección 1x1)
        if in_ch != out_ch:
            self.res = AnalogConv1d(in_ch, out_ch, 1,
                                    analog_levels=analog_levels, noise_std=noise_std)
        else:
            self.res = nn.Identity()

    def forward(self, x, knobs):
        res = self.res(x)
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]

        x = self.film(x, knobs)
        x = self.norm(x)
        return self.act(x + res)

# --- Modelo Principal TCN Analógico ---

class AnalogTCNModel(nn.Module):
    def __init__(self, input_size=10, channels=128, num_layers=12, kernel_size=3,
                 analog_levels=32, noise_std=0.02):
        super().__init__()

        self.channels = channels
        self.num_layers = num_layers
        self.num_control = input_size - 1

        self.layers = nn.ModuleList()

        # Primera capa (Input -> Channels)
        self.layers.append(
            AnalogTemporalBlock(1, channels, kernel_size, dilation=1,
                                knob_dim=self.num_control,
                                analog_levels=analog_levels, noise_std=noise_std)
        )

        # Capas profundas
        for i in range(1, num_layers):
            dilation = 2 ** i
            self.layers.append(
                AnalogTemporalBlock(channels, channels, kernel_size, dilation,
                                    knob_dim=self.num_control,
                                    analog_levels=analog_levels, noise_std=noise_std)
            )

        # Output Layer (1x1 Conv Analógica)
        self.output = AnalogConv1d(channels, 1, kernel_size=1, bias=False,
                                   analog_levels=analog_levels, noise_std=noise_std)

    def forward(self, x):
        audio = x[:, :, 0:1]
        control = x[:, :, 1:]

        audio = audio.permute(0, 2, 1) # (B, 1, T)
        knobs = control[:, 0, :]       # (B, 9)

        out = audio
        for layer in self.layers:
            out = layer(out, knobs)

        out = self.output(out)
        out = out.permute(0, 2, 1) # (B, T, 1)

        return out
    
class AnalogRNNModel(nn.Module):
    def __init__(self, input_size=10, hidden_size=256, num_layers=3,
                 analog_levels=32, noise_std=0.02):
        super(AnalogRNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.context_dim = input_size - 1
        self.noise_std = noise_std

        # AnalogLinear es seguro
        self.context_mlp = nn.Sequential(
            AnalogLinear(self.context_dim, 64, analog_levels=analog_levels, noise_std=noise_std),
            nn.ReLU(),
            AnalogLinear(64, 32, analog_levels=analog_levels, noise_std=noise_std),
            nn.Tanh()
        )

        self.rnn = nn.RNN(
            input_size=1 + 32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=0.2 if num_layers > 1 else 0
        )

        self.head = nn.Sequential(
            AnalogLinear(hidden_size, 128, analog_levels=analog_levels, noise_std=noise_std),
            nn.GELU(),
            AnalogLinear(128, 1, analog_levels=analog_levels, noise_std=noise_std)
        )

    def forward(self, x):
        # 1. Procesar contexto (AnalogLinear maneja su propio ruido)
        audio = x[:, :, 0:1]
        raw_context = x[:, 0, 1:]
        ctx_emb = self.context_mlp(raw_context)
        seq_len = audio.size(1)
        ctx_emb_expanded = ctx_emb.unsqueeze(1).repeat(1, seq_len, 1)
        rnn_input = torch.cat([audio, ctx_emb_expanded], dim=2)

        # 2. Ejecutar RNN (Con inyecciÃ³n segura Stateless)
        if self.training and self.noise_std > 0:
            noisy_params = {}
            for name, param in self.rnn.named_parameters():
                noisy_params[name] = param + torch.randn_like(param) * self.noise_std

            rnn_out, _ = stateless.functional_call(self.rnn, noisy_params, rnn_input)
        else:
            rnn_out, _ = self.rnn(rnn_input)

        # 3. Output
        output = self.head(rnn_out)

        return output