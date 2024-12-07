from abc import ABCMeta
from collections.abc import Callable

from modules.module.AestheticScoreModel import AestheticScoreModel
from modules.module.HPSv2ScoreModel import HPSv2ScoreModel
from modules.util.config.TrainConfig import TrainConfig
from modules.util.DiffusionScheduleCoefficients import DiffusionScheduleCoefficients
from modules.util.enum.AlignPropLoss import AlignPropLoss
from modules.util.enum.LossScaler import LossScaler
from modules.util.enum.LossWeight import LossWeight
from modules.util.loss.masked_loss import masked_losses
from modules.util.loss.vb_loss import vb_losses

import torch
import torch.nn.functional as F
from torch import Tensor


class ModelSetupDiffusionLossMixin(metaclass=ABCMeta):
    __coefficients: DiffusionScheduleCoefficients | None
    __alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None

    def __init__(self):
        super().__init__()
        self.__align_prop_loss_fn = None
        self.__coefficients = None
        self.__alphas_cumprod_fun = None

    def __align_prop_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
    ):
        if self.__align_prop_loss_fn is None:
            dtype = data['predicted'].dtype

            match config.align_prop_loss:
                case AlignPropLoss.HPS:
                    self.__align_prop_loss_fn = HPSv2ScoreModel(dtype)
                case AlignPropLoss.AESTHETIC:
                    self.__align_prop_loss_fn = AestheticScoreModel()

            self.__align_prop_loss_fn.to(device=train_device, dtype=dtype)
            self.__align_prop_loss_fn.requires_grad_(False)
            self.__align_prop_loss_fn.eval()

        losses = 0

        match config.align_prop_loss:
            case AlignPropLoss.HPS:
                with torch.autocast(device_type=train_device.type, dtype=data['predicted'].dtype):
                    losses = self.__align_prop_loss_fn(data['predicted'], batch['prompt'], train_device)
            case AlignPropLoss.AESTHETIC:
                losses = self.__align_prop_loss_fn(data['predicted'])

        return losses * config.align_prop_weight

    def __log_cosh_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
    ):
        diff = pred - target
        loss = diff + torch.nn.functional.softplus(-2.0*diff) - torch.log(torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device))
        return loss

    def __rational_quadratic_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            k: float,
    ):
        diff = pred - target
        squared_diff = diff * diff
        loss = squared_diff / (1.0/k + squared_diff)
        return loss

    def __smoothing_sigmoid_loss(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            k: float,
    ):
        diff = pred - target
        loss = diff + ( torch.nn.functional.softplus(-2.0*k*diff) - torch.log(torch.full(size=diff.size(), fill_value=2.0, dtype=torch.float32, device=diff.device)) ) / k
        return loss

    def __masked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ):
        losses = 0

        # guard for infinite or NaN values in predicted
        if torch.isinf(data['predicted']).any() or torch.isnan(data['predicted']).any():
            print(f"Warning: Infinite or NaN values detected in predicted - inf: {torch.isinf(data['predicted']).sum()}, nan: {torch.isnan(data['predicted']).sum()}")

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += masked_losses(
                losses=F.mse_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += masked_losses(
                losses=F.l1_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    reduction='none'
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * config.mae_strength

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += masked_losses(
                losses=self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32)
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * config.log_cosh_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data and self.__coefficients is not None:
            losses += masked_losses(
                losses=vb_losses(
                    coefficients=self.__coefficients,
                    x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                    x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                    t=data['timestep'],
                    predicted_eps=data['predicted'].to(dtype=torch.float32),
                    predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * config.vb_loss_strength

        # Rational Quadratic Loss
        if config.rational_quadratic_strength != 0:
            losses += masked_losses(
                losses=self.__rational_quadratic_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    config.rational_quadratic_k,
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * config.rational_quadratic_strength

        # Smoothing Sigmoid Loss
        if config.smoothing_sigmoid_strength != 0:
            losses += masked_losses(
                losses=self.__smoothing_sigmoid_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32),
                    config.smoothing_sigmoid_k,
                ),
                mask=batch['latent_mask'].to(dtype=torch.float32),
                unmasked_weight=config.unmasked_weight,
                normalize_masked_area_loss=config.normalize_masked_area_loss,
            ).mean([1, 2, 3]) * config.smoothing_sigmoid_strength

        # guard for infinite or NaN values in losses
        if losses.isnan().any() or losses.isinf().any():
            print(f"Warning: NaN or Infinite values detected in losses - inf: {losses.isinf().sum()}, nan: {losses.isnan().sum()}")
            print(f"predicted - inf: {torch.isinf(data['predicted']).sum()}, nan: {torch.isnan(data['predicted']).sum()}")
            print(f"predicted max: {data['predicted'].max()}, min: {data['predicted'].min()}")
            raise ValueError("NaN or Infinite values detected in losses")

        return losses

    def __unmasked_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
    ):
        losses = 0

        # guard for infinite or NaN values in predicted
        if torch.isinf(data['predicted']).any() or torch.isnan(data['predicted']).any():
            print(f"Warning: Infinite or NaN values detected in predicted - inf: {torch.isinf(data['predicted']).sum()}, nan: {torch.isnan(data['predicted']).sum()}")

        # MSE/L2 Loss
        if config.mse_strength != 0:
            losses += F.mse_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean([1, 2, 3]) * config.mse_strength

        # MAE/L1 Loss
        if config.mae_strength != 0:
            losses += F.l1_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                reduction='none'
            ).mean([1, 2, 3]) * config.mae_strength

        # log-cosh Loss
        if config.log_cosh_strength != 0:
            losses += self.__log_cosh_loss(
                    data['predicted'].to(dtype=torch.float32),
                    data['target'].to(dtype=torch.float32)
                ).mean([1, 2, 3]) * config.log_cosh_strength

        # VB loss
        if config.vb_loss_strength != 0 and 'predicted_var_values' in data:
            losses += vb_losses(
                coefficients=self.__coefficients,
                x_0=data['scaled_latent_image'].to(dtype=torch.float32),
                x_t=data['noisy_latent_image'].to(dtype=torch.float32),
                t=data['timestep'],
                predicted_eps=data['predicted'].to(dtype=torch.float32),
                predicted_var_values=data['predicted_var_values'].to(dtype=torch.float32),
            ).mean([1, 2, 3]) * config.vb_loss_strength

        # Rational Quadratic Loss
        if config.rational_quadratic_strength != 0:
            losses += self.__rational_quadratic_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                config.rational_quadratic_k,
            ).mean([1, 2, 3]) * config.rational_quadratic_strength

        # Smoothing Sigmoid Loss
        if config.smoothing_sigmoid_strength != 0:
            losses += self.__smoothing_sigmoid_loss(
                data['predicted'].to(dtype=torch.float32),
                data['target'].to(dtype=torch.float32),
                config.smoothing_sigmoid_k,
            ).mean([1, 2, 3]) * config.smoothing_sigmoid_strength

        if config.masked_training and config.normalize_masked_area_loss:
            clamped_mask = torch.clamp(batch['latent_mask'], config.unmasked_weight, 1)
            mask_mean = clamped_mask.mean(dim=(1, 2, 3))
            losses /= mask_mean

        # guard for infinite or NaN values in losse
        if losses.isnan().any() or losses.isinf().any():
            print(f"Warning: NaN or Infinite values detected in losses - inf: {losses.isinf().sum()}, nan: {losses.isnan().sum()}")
            print(f"predicted - inf: {torch.isinf(data['predicted']).sum()}, nan: {torch.isnan(data['predicted']).sum()}")
            print(f"predicted max: {data['predicted'].max()}, min: {data['predicted'].min()}")
            raise ValueError("NaN or Infinite values detected in losses")

        return losses

    def __snr(self, timesteps: Tensor, device: torch.device):
        if self.__coefficients:
            all_snr = (self.__coefficients.sqrt_alphas_cumprod /
                       self.__coefficients.sqrt_one_minus_alphas_cumprod) ** 2
            all_snr.to(device)
            snr = all_snr[timesteps]
        else:
            alphas_cumprod = self.__alphas_cumprod_fun(timesteps, 1)
            snr = alphas_cumprod / (1.0 - alphas_cumprod)

        return snr


    def __min_snr_weight(
            self,
            timesteps: Tensor,
            gamma: float,
            v_prediction: bool,
            device: torch.device
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
        # Denominator of the snr_weight increased by 1 if v-prediction is being used.
        if v_prediction:
            snr += 1.0
        snr_weight = (min_snr_gamma / snr).to(device)
        return snr_weight

    def __debiased_estimation_weight(
        self,
        timesteps: Tensor,
        v_prediction: bool,
        device: torch.device
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        weight = snr
        # The line below is a departure from the original paper.
        # This is to match the Kohya implementation, see: https://github.com/kohya-ss/sd-scripts/pull/889
        # In addition, it helps avoid numerical instability.
        torch.clip(weight, max=1.0e3, out=weight)
        if v_prediction:
            weight += 1.0
        torch.rsqrt(weight, out=weight)
        return weight

    def __p2_loss_weight(
        self,
        timesteps: Tensor,
        gamma: float,
        v_prediction: bool,
        device: torch.device,
    ) -> Tensor:
        snr = self.__snr(timesteps, device)
        if v_prediction:
            snr += 1.0
        return (1.0 + snr) ** -gamma

    def __sigma_weight(
            self,
            sigmas: Tensor,
            config: TrainConfig,
            timesteps: Tensor,
    ) -> Tensor:
        return sigmas[timesteps]

    def __logit_normal_weight(
            self,
            sigmas: Tensor,
            config: TrainConfig,
            timesteps: Tensor,
    ) -> Tensor:
        x = sigmas[timesteps]

        x = torch.clamp(x, min=1e-5, max=1-1e-5)

        logit_x = torch.log(x / (1 - x))

        mean = config.logit_normal_mean
        std = config.logit_normal_std

        # 1/std√2pi・1/x(1-x)・e^(-(logit(x)-mean)^2/2s^2)
        norm_const = 1.0 / (std * torch.sqrt(torch.tensor(2.0 * torch.pi)))
        jacobian = 1.0 / (x * (1.0 - x))
        exp_term = torch.exp(-0.5 * ((logit_x - mean) / std) ** 2)

        return norm_const * jacobian * exp_term

    def _diffusion_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            betas: Tensor | None = None,
            alphas_cumprod_fun: Callable[[Tensor, int], Tensor] | None = None,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        batch_size_scale = \
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION] \
                else config.batch_size
        gradient_accumulation_steps_scale = \
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.BATCH] \
                else config.gradient_accumulation_steps

        if self.__coefficients is None and betas is not None:
            self.__coefficients = DiffusionScheduleCoefficients.from_betas(betas)

        self.__alphas_cumprod_fun = alphas_cumprod_fun

        if data['loss_type'] == 'align_prop':
            losses = self.__align_prop_losses(batch, data, config, train_device)
        else:
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * batch_size_scale * gradient_accumulation_steps_scale

        losses *= loss_weight.to(device=losses.device, dtype=losses.dtype)

        # Apply timestep based loss weighting.
        if 'timestep' in data and data['loss_type'] != 'align_prop':
            v_pred = data.get('prediction_type', '') == 'v_prediction'
            match config.loss_weight_fn:
                case LossWeight.MIN_SNR_GAMMA:
                    losses *= self.__min_snr_weight(data['timestep'], config.loss_weight_strength, v_pred, losses.device)
                case LossWeight.DEBIASED_ESTIMATION:
                    losses *= self.__debiased_estimation_weight(data['timestep'], v_pred, losses.device)
                case LossWeight.P2:
                    losses *= self.__p2_loss_weight(data['timestep'], config.loss_weight_strength, v_pred, losses.device)

        return losses

    def _flow_matching_losses(
            self,
            batch: dict,
            data: dict,
            config: TrainConfig,
            train_device: torch.device,
            sigmas: Tensor | None = None,
    ) -> Tensor:
        loss_weight = batch['loss_weight']
        batch_size_scale = \
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.GRADIENT_ACCUMULATION] \
                else config.batch_size
        gradient_accumulation_steps_scale = \
            1 if config.loss_scaler in [LossScaler.NONE, LossScaler.BATCH] \
                else config.gradient_accumulation_steps

        if data['loss_type'] == 'align_prop':
            losses = self.__align_prop_losses(batch, data, config, train_device)
        else:
            # TODO: don't disable masked loss functions when has_conditioning_image_input is true.
            #  This breaks if only the VAE is trained, but was loaded from an inpainting checkpoint
            if config.masked_training and not config.model_type.has_conditioning_image_input():
                losses = self.__masked_losses(batch, data, config)
            else:
                losses = self.__unmasked_losses(batch, data, config)

        # apply loss weighting
        if sigmas is not None and 'timestep' in data:
            match config.loss_weight_fn:
                case LossWeight.SIGMA:
                    losses *= self.__sigma_weight(sigmas.flip(0), config, data['timestep'])
                case LossWeight.LOGIT_NORMAL:
                    losses *= self.__logit_normal_weight(sigmas.flip(0), config, data['timestep'])

        # Scale Losses by Batch and/or GA (if enabled)
        losses = losses * batch_size_scale * gradient_accumulation_steps_scale

        losses *= loss_weight.to(device=losses.device, dtype=losses.dtype)

        return losses
