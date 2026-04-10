#!/usr/bin/env python3
"""
Benchmark script to compare fused-SSIM vs PyTorch SSIM performance during training.

This script runs the same training pipeline with two configurations:
1. Using standard PyTorch SSIM implementation
2. Using the fused-SSIM CUDA kernel implementation

It measures:
- Per-iteration training time (excluding warm-up)
- Total training time for the benchmark
- GPU memory usage

Usage:
    python benchmark_fused_ssim.py --num_iterations 100
    
Optional flags:
    --num_iterations: Number of training iterations to run (default: 100)
    --warmup_iterations: Number of warm-up iterations before timing (default: 10)
    --output_dir: Output directory for results (default: ./benchmark_results)
"""

import os
import json
import torch
import time
import subprocess
import sys
from random import randint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from functools import partial
from scene.gaussian_model import build_scaling_rotation
from solver.gaussian_model_vector import GaussianModelVector
from solver.adam_optimizer import AdamOptimizer
from solver.sophia_optimizer import SophiaOptimizer
from solver.solver_functions import construct_loss_func, construct_g_func, construct_JTJv_func, dot, saxpy, construct_Dhat_func
from solver.hellinger_clip import clip_hellinger
from solver.uniform_clip import clip_uniform

try:
    from fused_ssim import fused_ssim, FusedSSIMMap
    FUSED_SSIM_AVAILABLE = True
    print("FUSED_SSIM_AVAILABLE: True")
except Exception as e:
    FUSED_SSIM_AVAILABLE = False
    print(f"FUSED_SSIM_AVAILABLE: False (Error: {e})")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


class SSIMWrapper:
    """Wrapper to toggle between standard and fused SSIM implementations"""
    
    def __init__(self, use_fused_ssim=False):
        self.use_fused_ssim = use_fused_ssim
        self.actually_used_fused = False
        self.call_count = 0
        self.total_time = 0.0
    
    def __call__(self, img1, img2):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        
        if self.use_fused_ssim and FUSED_SSIM_AVAILABLE:
            try:
                C1 = 0.01 ** 2
                C2 = 0.03 ** 2
                # Use fused SSIM implementation
                ssim_map = fused_ssim(C1, C2, img1, img2, padding="same", train=True)
                result = ssim_map.mean()
                self.actually_used_fused = True
            except Exception as e:
                print(f"Warning: Fused SSIM call failed: {e}, falling back to PyTorch implementation")
                result = ssim(img1, img2)
                self.actually_used_fused = False
        else:
            result = ssim(img1, img2)
            self.actually_used_fused = False
        
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0  # Convert to seconds
        self.total_time += elapsed
        self.call_count += 1
        
        return result


def benchmark_training(dataset, opt, pipe, num_iterations, warmup_iterations, 
                       use_fused_ssim, output_dir):
    """
    Run a benchmark of the training pipeline.
    
    Args:
        dataset: ModelParams with dataset configuration
        opt: OptimizationParams
        pipe: PipelineParams
        num_iterations: Number of iterations to benchmark
        warmup_iterations: Number of warm-up iterations before timing
        use_fused_ssim: Whether to use fused SSIM
        output_dir: Directory to save benchmark results
    """
    
    print(f"\n{'='*80}")
    print(f"Benchmark Configuration:")
    print(f"  Mode: {'FUSED_SSIM' if use_fused_ssim else 'PyTorch SSIM'}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Dataset: {dataset.source_path}")
    print(f"{'='*80}\n")

    train_test_exp = False
    
    # Set up output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model and scene
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    kl_threshold_func = get_expon_lr_func(lr_init=opt.kl_threshold_init, 
                                          lr_final=opt.kl_threshold_final, 
                                          lr_delay_mult=opt.kl_threshold_delay_mult,
                                          max_steps=opt.iterations)
    
    # Initialize optimizers
    lr = GaussianModelVector(
        xyz=opt.xyz_lr_init,
        features_dc=opt.features_dc_lr,
        features_rest=opt.features_rest_lr,
        scaling=opt.scaling_lr,
        rotation=opt.rotation_lr,
        opacity=opt.opacity_lr,
        exposure=opt.exposure_lr,
        gaussians=gaussians
    )
    adam_optimizer = AdamOptimizer(lr=lr, betas=(opt.adam_beta1, opt.adam_beta2), eps=1e-15, clip=False)
    adam_optimizer.reset()
    
    sophia_optimizer = SophiaOptimizer(
        lr=lr,
        betas=(opt.adahessian_beta1, opt.adahessian_beta2),
        eps=1e-20, clip=False,
        gamma=opt.sophia_gamma,
        diagonal_update_interval=opt.diagonal_update_interval,
        num_init_iter=opt.diagonal_init_iter,
        num_init_restart_iter=opt.diagonal_init_restart_iter,
        num_update_iter=opt.diagonal_update_iter,
        num_update_restart_iter=opt.diagonal_update_restart_iter,
        diagonal_accum_abs=opt.diagonal_accum_abs,
        diagonal_adam_precondition=opt.diagonal_adam_precondition,
    )
    sophia_optimizer.reset()
    
    # Clipping function (from train_mcmc_sophia_hellinger.py)
    clip_func = clip_uniform if opt.tr_func == "uniform" else clip_hellinger
    
    # Create SSIM wrapper
    ssim_wrapper = SSIMWrapper(use_fused_ssim=use_fused_ssim)
    
    # Timing statistics
    iteration_times = []
    total_start = time.time()
    
    print(f"Starting benchmark with {num_iterations + warmup_iterations} iterations "
          f"({warmup_iterations} warmup + {num_iterations} timed)...\n")
    
    pbar = tqdm(range(warmup_iterations + num_iterations), desc="Training benchmark")
    
    viewpoint_stack = None
    
    for iteration in range(warmup_iterations + num_iterations):
        iter_start = time.time()
        
        # Update learning rates
        xyz_lr = gaussians.update_learning_rate(iteration)
        lr = GaussianModelVector(
            xyz=xyz_lr,
            features_dc=opt.features_dc_lr,
            features_rest=opt.features_rest_lr,
            scaling=opt.scaling_lr,
            rotation=opt.rotation_lr,
            opacity=opt.opacity_lr,
            exposure=opt.exposure_lr,
            gaussians=gaussians
        )
        adam_optimizer.update_lr(lr)
        
        # Every 1000 iterations, increase SH degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # Render
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        
        # Compute loss using appropriate SSIM implementation
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        # This is where we use the wrapped SSIM function
        ssim_value = ssim_wrapper(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # Backward pass
        loss.backward()
        g = GaussianModelVector.from_gaussians_grad(gaussians)
        
        # Construct loss functions (from train_mcmc_sophia_hellinger.py)
        render_args = {"iteration": iteration,
                       "opt": opt,
                       "pipe": pipe,
                       "bg": bg,
                       "train_test_exp": train_test_exp,
                       "depth_l1_weight": depth_l1_weight,
                       "loss_type": opt.loss_type,
                       "huber_delta": opt.huber_delta,
                       "disable_ssim": opt.disable_ssim,
                       "batch_size": 1,
                       "pixel_mask": None,
                       }
        
        loss_func = construct_loss_func(**render_args)
        g_func = construct_g_func(**render_args)
        JTJv_func = construct_JTJv_func(**render_args)
        Dhat_func = construct_Dhat_func(**render_args)
        z_gen_func = partial(GaussianModelVector.rademacher_like, gaussians)
        
        JTJv_func1 = partial(JTJv_func, gaussians=gaussians, viewpoint_cams=[viewpoint_cam], S=None, scale=1)
        Dhat_func1 = partial(Dhat_func, gaussians=gaussians, viewpoint_cams=[viewpoint_cam])
        
        # Optimizer steps
        if iteration < warmup_iterations + num_iterations - 1:
            s_adam = adam_optimizer.get_update(g)
            s_sophia = sophia_optimizer.get_update(g, JTJv_func1, Dhat_func1, z_gen_func, S=None)
            
            # Use Adam for this benchmark
            s = s_sophia
            gaussians.update_step(s)
            gaussians.optimizer.zero_grad(set_to_none=True)
        
        iter_end = time.time()
        iter_time = iter_end - iter_start
        
        # Only record timing after warmup
        if iteration >= warmup_iterations:
            iteration_times.append(iter_time)
        
        pbar.update(1)
        if iteration < warmup_iterations:
            pbar.set_postfix({"status": f"warmup ({iteration + 1}/{warmup_iterations})"})
        else:
            avg_time = sum(iteration_times) / len(iteration_times)
            pbar.set_postfix({"avg_iter_time": f"{avg_time:.4f}s"})
    
    pbar.close()
    
    total_end = time.time()
    total_time = total_end - total_start
    
    # Compute statistics
    avg_iter_time = sum(iteration_times) / len(iteration_times)
    min_iter_time = min(iteration_times)
    max_iter_time = max(iteration_times)
    
    # Results
    results = {
        "mode": "fused_ssim" if use_fused_ssim else "pytorch_ssim",
        "requested_fused_ssim": use_fused_ssim,
        "actually_used_fused": ssim_wrapper.actually_used_fused and FUSED_SSIM_AVAILABLE,
        "fused_ssim_available": FUSED_SSIM_AVAILABLE,
        "num_iterations": num_iterations,
        "warmup_iterations": warmup_iterations,
        "total_time_seconds": total_time,
        "avg_iter_time_seconds": avg_iter_time,
        "min_iter_time_seconds": min_iter_time,
        "max_iter_time_seconds": max_iter_time,
        "iterations_per_second": 1.0 / avg_iter_time,
        "ssim_call_count": ssim_wrapper.call_count,
        "ssim_total_time_seconds": ssim_wrapper.total_time,
        "ssim_avg_time_ms": (ssim_wrapper.total_time / ssim_wrapper.call_count * 1000) if ssim_wrapper.call_count > 0 else 0,
    }
    
    # Add GPU memory stats
    try:
        mem_smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode("utf-8").strip()
        results["gpu_memory_used_mb"] = float(mem_smi)
    except:
        results["gpu_memory_used_mb"] = None
    
    try:
        mem_torch = torch.cuda.max_memory_allocated() / (1024 ** 2)
        results["torch_max_memory_allocated_mb"] = mem_torch
    except:
        results["torch_max_memory_allocated_mb"] = None
    
    # Print results
    print(f"\n{'='*80}")
    print(f"Benchmark Results: {'FUSED_SSIM' if use_fused_ssim else 'PyTorch SSIM'}")
    print(f"{'='*80}")
    if use_fused_ssim and not FUSED_SSIM_AVAILABLE:
        print(f"⚠️  WARNING: Fused SSIM was requested but not available! Using PyTorch fallback.")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Average Iteration Time: {avg_iter_time:.6f}s")
    print(f"Min Iteration Time: {min_iter_time:.6f}s")
    print(f"Max Iteration Time: {max_iter_time:.6f}s")
    print(f"Iterations per Second: {results['iterations_per_second']:.2f}")
    print(f"SSIM Total Time: {ssim_wrapper.total_time:.4f}s ({ssim_wrapper.total_time/total_time*100:.1f}% of total)")
    print(f"SSIM Avg per Call: {results['ssim_avg_time_ms']:.3f}ms")
    if results["gpu_memory_used_mb"]:
        print(f"GPU Memory Used: {results['gpu_memory_used_mb']:.1f} MB")
    if results["torch_max_memory_allocated_mb"]:
        print(f"Torch Max Memory: {results['torch_max_memory_allocated_mb']:.1f} MB")
    print(f"{'='*80}\n")
    
    return results


def main():
    parser = ArgumentParser(description="Benchmark fused-SSIM vs PyTorch SSIM")
    
    # Original model/optimization/pipeline params
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # Benchmark-specific params
    parser.add_argument("--num_iterations", type=int, default=100,
                       help="Number of iterations to benchmark (excluding warmup)")
    parser.add_argument("--warmup_iterations", type=int, default=10,
                       help="Number of warmup iterations before timing")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results",
                       help="Directory to save benchmark results")
    parser.add_argument("--run_both", action="store_true", default=False,
                       help="Run both fused and pytorch versions and compare")
    
    args = parser.parse_args(sys.argv[1:])
    
    # Setup
    safe_state(silent=False)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract parameters
    dataset_params = lp.extract(args)
    opt_params = op.extract(args)
    pipe_params = pp.extract(args)
    
    all_results = {}
    
    # Run benchmark
    if args.run_both:
        print("\n" + "="*80)
        print("Running benchmark with BOTH PyTorch and Fused SSIM implementations")
        print("="*80)
        
        # Run PyTorch version
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        results_pytorch = benchmark_training(
            dataset_params, opt_params, pipe_params,
            args.num_iterations, args.warmup_iterations,
            use_fused_ssim=False,
            output_dir=args.output_dir
        )
        all_results["pytorch_ssim"] = results_pytorch
        
        # Run Fused version
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        results_fused = benchmark_training(
            dataset_params, opt_params, pipe_params,
            args.num_iterations, args.warmup_iterations,
            use_fused_ssim=True,
            output_dir=args.output_dir
        )
        all_results["fused_ssim"] = results_fused
        
        # Comparison
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Check if fused was actually used
        if results_fused["requested_fused_ssim"] and not results_fused["actually_used_fused"]:
            print("⚠️  WARNING: Fused SSIM was requested but not available!")
            print("   Both benchmarks may be using PyTorch SSIM. Check build configuration.")
            print("="*80 + "\n")
            return all_results
        
        pytorch_time = results_pytorch["avg_iter_time_seconds"]
        fused_time = results_fused["avg_iter_time_seconds"]
        speedup = pytorch_time / fused_time
        improvement = (pytorch_time - fused_time) / pytorch_time * 100
        
        print(f"PyTorch SSIM - Avg Iteration Time: {pytorch_time:.6f}s ({results_pytorch['iterations_per_second']:.2f} it/s)")
        print(f"Fused SSIM   - Avg Iteration Time: {fused_time:.6f}s ({results_fused['iterations_per_second']:.2f} it/s)")
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Time Improvement: {improvement:.1f}%")
        if improvement < 0:
            print(f"WARNING: Fused SSIM is SLOWER by {abs(improvement):.1f}%")
        else:
            print(f"✓ Fused SSIM is {improvement:.1f}% faster")
        print("="*80 + "\n")
        
        all_results["comparison"] = {
            "speedup": speedup,
            "time_improvement_percent": improvement,
            "pytorch_avg_iter_time": pytorch_time,
            "fused_avg_iter_time": fused_time,
        }
    
    else:
        # Run single version
        use_fused = True if args.run_both else (args.run_both == False and FUSED_SSIM_AVAILABLE)
        results = benchmark_training(
            dataset_params, opt_params, pipe_params,
            args.num_iterations, args.warmup_iterations,
            use_fused_ssim=use_fused,
            output_dir=args.output_dir
        )
        all_results["benchmark"] = results
    
    # Save results to JSON
    results_file = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()
