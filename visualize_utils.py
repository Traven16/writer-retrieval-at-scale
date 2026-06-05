from __future__ import annotations

import os
import tempfile
from typing import Optional, Sequence, Tuple

import torch
import numpy as np
from PIL import Image

from data.dataset import normalize_patch_tensor


def _patch_to_image(patch: torch.Tensor) -> torch.Tensor:
    if patch.dim() != 3:
        raise ValueError(f"Expected patch shape (C,H,W), got {patch.shape}")
    patch = patch.detach().cpu().float()
    patch = patch.permute(1, 2, 0)
    min_val = patch.min()
    max_val = patch.max()
    if (max_val - min_val) > 1e-6:
        patch = (patch - min_val) / (max_val - min_val)
    patch = patch.clamp(0.0, 1.0)
    return patch


def _save_heatmap(
    mat: torch.Tensor,
    out_path: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_patches: Optional[Sequence[torch.Tensor]] = None,
    y_patches: Optional[Sequence[torch.Tensor]] = None,
    x_labels: Optional[Sequence[str]] = None,
    normalize: bool = False,
    x_strip: Optional[torch.Tensor] = None,
    annotate: bool = False,
    cmap: str = "gray",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - best-effort visualization
        raise RuntimeError(f"matplotlib is required for visualization ({exc})") from exc

    if normalize:
        mat = mat / max(mat.max().item(), 1e-8)
    show_x = bool(x_patches)
    show_y = bool(y_patches)
    show_hist = x_strip is not None
    patch_px = 32
    try:
        if show_x and x_patches:
            patch_px = int(x_patches[0].shape[-1])
        elif show_y and y_patches:
            patch_px = int(y_patches[0].shape[-1])
    except Exception:
        patch_px = 32
    x_scale = patch_px if (show_x or x_labels is not None or show_hist) else 1
    y_scale = patch_px if (show_y or show_x or x_labels is not None) else 1
    if show_x or show_y or show_hist:
        if show_x and x_patches and mat.size(1) != len(x_patches):
            raise ValueError(
                f"Heatmap width ({mat.size(1)}) does not match x_patches ({len(x_patches)})"
            )
        if show_y and y_patches and mat.size(0) != len(y_patches):
            raise ValueError(
                f"Heatmap height ({mat.size(0)}) does not match y_patches ({len(y_patches)})"
            )
        from matplotlib import gridspec

        top_h = 1 if show_x else 0
        hist_h = 1 if show_hist else 0
        left_w = 1 if show_y else 0
        mat_w_px = int(mat.size(1)) * x_scale
        mat_h_px = int(mat.size(0)) * y_scale
        fig_w_px = mat_w_px + (patch_px if show_y else 0)
        fig_h_px = mat_h_px + (patch_px if show_x else 0) + (patch_px if show_hist else 0)
        dpi = 100
        fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)
        gs = gridspec.GridSpec(
            3,
            2,
            width_ratios=[
                patch_px if show_y else 0.01,
                mat_w_px,
            ],
            height_ratios=[
                patch_px if show_hist else 0.01,
                patch_px if show_x else 0.01,
                mat_h_px,
            ],
            wspace=0.0,
            hspace=0.0,
        )
        ax = fig.add_subplot(gs[2, 1])
        ax.imshow(
            mat,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
            extent=(0, mat_w_px, mat_h_px, 0),
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_axis_off()
        if annotate:
            mat_np = mat.detach().cpu().numpy()
            rows, cols = mat_np.shape
            for i in range(rows):
                for j in range(cols):
                    ax.text(
                        (j + 0.5) * x_scale,
                        (i + 0.5) * y_scale,
                        f"{mat_np[i, j] * 100:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="red",
                    )
        if show_hist:
            ax_hist = fig.add_subplot(gs[0, 1])
            strip = x_strip.detach().cpu().float()
            strip = strip / max(strip.max().item(), 1e-8)
            ax_hist.imshow(
                strip.view(1, -1),
                aspect="auto",
                interpolation="nearest",
                extent=(0, mat_w_px, patch_px, 0),
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
            )
            ax_hist.axis("off")
        if show_x:
            ax_top = fig.add_subplot(gs[1, 1])
            patches = [_patch_to_image(patch).numpy() for patch in x_patches]
            strip = torch.from_numpy(patches[0])
            for patch in patches[1:]:
                strip = torch.cat([strip, torch.from_numpy(patch)], dim=1)
            ax_top.imshow(strip, aspect="auto", interpolation="nearest", extent=(0, mat_w_px, patch_px, 0))
            ax_top.axis("off")
        if show_y:
            ax_left = fig.add_subplot(gs[2, 0])
            patches = [_patch_to_image(patch).numpy() for patch in y_patches]
            strip = torch.from_numpy(patches[0])
            for patch in patches[1:]:
                strip = torch.cat([strip, torch.from_numpy(patch)], dim=0)
            ax_left.imshow(strip, aspect="auto", interpolation="nearest", extent=(0, patch_px, mat_h_px, 0))
            ax_left.axis("off")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        fig, ax = plt.subplots(figsize=(max(8, mat.size(1) * 0.2), max(6, mat.size(0) * 0.2)))
        ax.imshow(mat, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if x_labels is not None:
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        else:
            ax.set_xticks([])
        ax.set_yticks([])
        if annotate:
            mat_np = mat.detach().cpu().numpy()
            rows, cols = mat_np.shape
            for i in range(rows):
                for j in range(cols):
                    ax.text(
                        j,
                        i,
                        f"{mat_np[i, j] * 100:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                        color="red",
                    )
    if not (show_x or show_y):
        fig.tight_layout()
    fig.savefig(out_path, dpi=fig.dpi)
    plt.close(fig)


def render_decoder_cross_attention_image(
    model,
    patches: torch.Tensor,
    sample_index: int = 0,
    annotate: bool = False,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Render decoder cross-attention/assignments for one sample as CHW float image."""
    if patches.dim() != 5:
        raise ValueError(f"Expected patches shape (B,N,C,H,W), got {patches.shape}")
    if sample_index < 0 or sample_index >= patches.size(0):
        raise ValueError(f"sample_index out of range: {sample_index} for batch {patches.size(0)}")

    if mean is None or std is None:
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)

    patches_norm = normalize_patch_tensor(patches, mean, std)
    with torch.no_grad():
        embeddings = model._encode_patches(patches_norm)

    b = sample_index
    patch_images = [patches[b, i].detach().cpu() for i in range(patches.size(1))]
    mat: Optional[torch.Tensor] = None
    y_label = "Head"
    title = f"Decoder cross-attention | batch {b}"
    x_strip: Optional[torch.Tensor] = None

    if model.decoder_type == "xvlad" and hasattr(model.decoder, "decoder"):
        xvlad = model.decoder.decoder
        if hasattr(xvlad, "assignment_heads") and hasattr(xvlad, "head_projections"):
            num_clusters = int(getattr(xvlad, "num_clusters", 1))
            per_head = []
            for head_index in range(len(xvlad.assignment_heads)):
                projected = xvlad.head_projections[head_index](embeddings)
                assignments = torch.softmax(
                    xvlad.assignment_heads[head_index](projected), dim=-1
                )
                keep = assignments[:, :, :num_clusters].transpose(1, 2).contiguous()
                per_head.append(keep.detach().cpu())
            if per_head:
                per_head_tensor = torch.stack(per_head, dim=1)  # (B, heads, clusters, patches)
                mat = per_head_tensor[b].reshape(-1, per_head_tensor.size(-1))
                x_strip = mat.mean(dim=0)
                x_strip = x_strip / x_strip.sum().clamp(min=1e-8)
                y_label = "Head" if num_clusters == 1 else "Head x Cluster"
                title = (
                    f"X-VLAD keep assignments (all heads) | batch {b}"
                    if num_clusters == 1
                    else f"X-VLAD keep assignments (all heads x clusters) | batch {b}"
                )
    elif model.decoder_type == "ghostvlad" and hasattr(model.decoder, "decoder"):
        ghostvlad = model.decoder.decoder
        if hasattr(ghostvlad, "assignment"):
            assignments = torch.softmax(ghostvlad.assignment(embeddings), dim=-1).detach().cpu()
            num_clusters = int(getattr(ghostvlad, "num_clusters", 1))
            non_ghost = assignments[:, :, :num_clusters].transpose(1, 2).contiguous()
            mat = non_ghost[b]
            x_strip = mat.mean(dim=0)
            x_strip = x_strip / x_strip.sum().clamp(min=1e-8)
            y_label = "Cluster"
            title = f"GhostVLAD non-ghost assignments | batch {b}"
    elif hasattr(model.decoder, "decoder") and getattr(model.decoder.decoder, "layers", None):
        layers = model.decoder.decoder.layers
        queries = model.decoder.query_tokens.expand(embeddings.size(0), -1, -1)
        output = queries
        if len(layers) > 1:
            for layer in layers[:-1]:
                output = layer(output, embeddings)
        dec_layer = layers[-1]
        tgt2 = dec_layer.self_attn(output, output, output, need_weights=False)[0]
        output = dec_layer.norm1(output + dec_layer.dropout1(tgt2))
        _, attn_weights = dec_layer.multihead_attn(
            output,
            embeddings,
            embeddings,
            need_weights=True,
            average_attn_weights=False,
        )
        cls_attn = attn_weights[:, :, 0, :].detach().cpu()  # (B, heads, patches)
        mat = cls_attn[b]
        x_strip = mat.mean(dim=0)
        x_strip = x_strip / x_strip.sum().clamp(min=1e-8)
        y_label = "Attention head"
        title = f"Decoder cross-attention (last layer, CLS only) | batch {b}"

    if mat is None:
        return None

    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        _save_heatmap(
            mat,
            tmp_path,
            title=title,
            xlabel="Patch index",
            ylabel=y_label,
            x_patches=patch_images,
            x_strip=x_strip,
            normalize=False,
            annotate=annotate,
        )
        img = Image.open(tmp_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def visualize_retrieval_attention(
    model,
    loader,
    device: torch.device,
    out_dir: str,
    annotate: bool = True,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    batch = next(iter(loader))
    patches = batch[0].to(device)
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "mean") and hasattr(dataset, "std"):
        mean, std = dataset.mean, dataset.std
    else:
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    patches_norm = normalize_patch_tensor(patches, mean, std)
    model.eval()
    with torch.no_grad():
        embeddings = model._encode_patches(patches_norm)
        batch_size, num_patches, _, height, width = patches.shape

        # Norm visualization (patch_proj vs encoder output)
        norm_dir = os.path.join(out_dir, "patch_norms")
        os.makedirs(norm_dir, exist_ok=True)
        # compute proj/enc for first batch only
        proj_first = []
        with torch.no_grad():
            flat = patches_norm.view(batch_size * num_patches, patches.size(2), height, width)
            feats = model.patch_embedding(flat)
            if isinstance(feats, (list, tuple)):
                feats = feats[-1]
            if feats.dim() == 4:
                feats = feats.mean(dim=(2, 3))
            elif feats.dim() == 3:
                feats = feats[:, 0, :]
            feats = feats.view(batch_size * num_patches, -1)
            proj_first = model.patch_proj(feats).view(batch_size, num_patches, -1).detach().cpu()
        enc_first = embeddings.detach().cpu()
        for b in range(batch_size):
            rows = num_patches
            cols = 4
            cell = 32
            grid = np.full((rows * cell, cols * cell, 3), 255, dtype=np.uint8)
            pnorms = torch.norm(proj_first[b], dim=1).cpu().numpy()
            enorms = torch.norm(enc_first[b], dim=1).cpu().numpy()
            dnorms = enorms - pnorms
            def _norm_to_color(vals, vmin=None, vmax=None, cmap="viridis"):
                import matplotlib.pyplot as plt
                if vmin is None:
                    vmin = float(np.min(vals))
                if vmax is None:
                    vmax = float(np.max(vals))
                norm = (vals - vmin) / max(vmax - vmin, 1e-6)
                cmap_fn = plt.get_cmap(cmap)
                colors = (cmap_fn(norm)[:, :3] * 255).astype(np.uint8)
                return colors
            p_colors = _norm_to_color(pnorms, cmap="viridis")
            e_colors = _norm_to_color(enorms, cmap="viridis")
            d_colors = _norm_to_color(dnorms, vmin=-np.max(np.abs(dnorms)), vmax=np.max(np.abs(dnorms)), cmap="coolwarm")
            for i in range(num_patches):
                patch_img = (_patch_to_image(patches[b, i]).numpy() * 255).astype(np.uint8)
                r0 = i * cell
                grid[r0 : r0 + cell, 0:cell] = patch_img
                pnorm = float(torch.norm(proj_first[b, i]).item())
                enorm = float(torch.norm(enc_first[b, i]).item())
                grid[r0 : r0 + cell, cell : 2 * cell] = p_colors[i]
                grid[r0 : r0 + cell, 2 * cell : 3 * cell] = e_colors[i]
                grid[r0 : r0 + cell, 3 * cell : 4 * cell] = d_colors[i]
                img1 = Image.fromarray(grid[r0 : r0 + cell, cell : 2 * cell])
                img2 = Image.fromarray(grid[r0 : r0 + cell, 2 * cell : 3 * cell])
                img3 = Image.fromarray(grid[r0 : r0 + cell, 3 * cell : 4 * cell])
                try:
                    from PIL import ImageDraw
                    draw1 = ImageDraw.Draw(img1)
                    draw2 = ImageDraw.Draw(img2)
                    draw3 = ImageDraw.Draw(img3)
                    draw1.text((2, 10), f"{pnorm:.2f}", fill=(0, 0, 0))
                    draw2.text((2, 10), f"{enorm:.2f}", fill=(0, 0, 0))
                    draw3.text((2, 10), f"{enorm - pnorm:.2f}", fill=(0, 0, 0))
                except Exception:
                    pass
                grid[r0 : r0 + cell, cell : 2 * cell] = np.array(img1)
                grid[r0 : r0 + cell, 2 * cell : 3 * cell] = np.array(img2)
                grid[r0 : r0 + cell, 3 * cell : 4 * cell] = np.array(img3)
            Image.fromarray(grid).save(os.path.join(norm_dir, f"patch_norms_b{b:02d}.png"))

        # Encoder self-attention (first layer)
        encoder_attn: Optional[torch.Tensor] = None
        if model.patch_encoder is not None and getattr(model.patch_encoder, "layers", None):
            enc_layer = model.patch_encoder.layers[0]
            attn_out = enc_layer.self_attn(
                embeddings,
                embeddings,
                embeddings,
                need_weights=True,
                average_attn_weights=False,
            )
            attn_weights = attn_out[1]  # (B, heads, N, N)
            encoder_attn = attn_weights.mean(dim=1).detach().cpu()
            agg_dir = os.path.join(out_dir, "encoder_self_attention", "aggregate")
            head_dir = os.path.join(out_dir, "encoder_self_attention", "per_head")
            os.makedirs(agg_dir, exist_ok=True)
            os.makedirs(head_dir, exist_ok=True)
            for b in range(encoder_attn.size(0)):
                patch_images = [patches[b, i].detach().cpu() for i in range(encoder_attn.size(1))]
                _save_heatmap(
                    encoder_attn[b],
                    os.path.join(agg_dir, f"encoder_self_attention_b{b:02d}.png"),
                    title=f"Encoder self-attention (layer 1, mean heads) | batch {b}",
                    xlabel="Patch index (key)",
                    ylabel="Patch index (query)",
                    x_patches=patch_images,
                    y_patches=patch_images,
                    annotate=annotate,
                )
                for h in range(attn_weights.size(1)):
                    _save_heatmap(
                        attn_weights[b, h].detach().cpu(),
                        os.path.join(head_dir, f"encoder_self_attention_b{b:02d}_h{h:02d}.png"),
                        title=f"Encoder self-attention (layer 1, head {h}) | batch {b}",
                        xlabel="Patch index (key)",
                        ylabel="Patch index (query)",
                        x_patches=patch_images,
                        y_patches=patch_images,
                        annotate=annotate,
                    )

        # Decoder cross-attention / assignment map
        decoder_attn: Optional[torch.Tensor] = None
        if model.decoder_type == "xvlad" and hasattr(model.decoder, "decoder"):
            xvlad = model.decoder.decoder
            if hasattr(xvlad, "assignment_heads") and hasattr(xvlad, "head_projections"):
                num_clusters = int(getattr(xvlad, "num_clusters", 1))
                per_head = []
                for head_index in range(len(xvlad.assignment_heads)):
                    projected = xvlad.head_projections[head_index](embeddings)
                    assignments = torch.softmax(
                        xvlad.assignment_heads[head_index](projected), dim=-1
                    )
                    keep = assignments[:, :, :num_clusters].transpose(1, 2).contiguous()
                    per_head.append(keep.detach().cpu())  # (B, clusters, patches)
                if per_head:
                    per_head_tensor = torch.stack(per_head, dim=1)  # (B, heads, clusters, patches)
                    agg = per_head_tensor.mean(dim=1)  # (B, clusters, patches)
                    decoder_attn = agg
                    for b in range(agg.size(0)):
                        patch_images = [patches[b, i].detach().cpu() for i in range(agg.size(2))]
                        # Show all heads at once in the default plot.
                        # For multiple clusters, each row is one (head, cluster) pair.
                        head_cluster = per_head_tensor[b].reshape(-1, per_head_tensor.size(-1))
                        hc_strip = head_cluster.mean(dim=0)
                        hc_strip = hc_strip / hc_strip.sum().clamp(min=1e-8)
                        _save_heatmap(
                            head_cluster,
                            os.path.join(out_dir, f"decoder_cross_attention_b{b:02d}.png"),
                            title=(
                                f"X-VLAD keep assignments (all heads) | batch {b}"
                                if num_clusters == 1
                                else f"X-VLAD keep assignments (all heads x clusters) | batch {b}"
                            ),
                            xlabel="Patch index",
                            ylabel="Head" if num_clusters == 1 else "Head x Cluster",
                            x_patches=patch_images,
                            x_strip=hc_strip,
                            normalize=False,
                            annotate=annotate,
                        )
        elif model.decoder_type == "ghostvlad" and hasattr(model.decoder, "decoder"):
            ghostvlad = model.decoder.decoder
            if hasattr(ghostvlad, "assignment"):
                assignments = torch.softmax(ghostvlad.assignment(embeddings), dim=-1).detach().cpu()
                num_clusters = int(getattr(ghostvlad, "num_clusters", 1))
                non_ghost = assignments[:, :, :num_clusters]  # (B, patches, non-ghost clusters)
                decoder_attn = non_ghost.transpose(1, 2).contiguous()  # (B, clusters, patches)
                for b in range(decoder_attn.size(0)):
                    patch_images = [patches[b, i].detach().cpu() for i in range(decoder_attn.size(2))]
                    if decoder_attn.size(1) == 1:
                        title = f"GhostVLAD non-ghost assignment | batch {b}"
                    else:
                        title = f"GhostVLAD non-ghost assignments | batch {b}"
                    _save_heatmap(
                        decoder_attn[b],
                        os.path.join(out_dir, f"decoder_cross_attention_b{b:02d}.png"),
                        title=title,
                        xlabel="Patch index",
                        ylabel="Cluster",
                        x_patches=patch_images,
                        normalize=False,
                        annotate=annotate,
                    )
        elif hasattr(model.decoder, "decoder") and getattr(model.decoder.decoder, "layers", None):
            layers = model.decoder.decoder.layers
            queries = model.decoder.query_tokens.expand(embeddings.size(0), -1, -1)
            output = queries
            if len(layers) > 1:
                for layer in layers[:-1]:
                    output = layer(output, embeddings)
            dec_layer = layers[-1]
            # Self-attention block
            tgt2 = dec_layer.self_attn(
                output, output, output, need_weights=False
            )[0]
            output = output + dec_layer.dropout1(tgt2)
            output = dec_layer.norm1(output)
            # Cross-attention block with weights
            tgt2, attn_weights = dec_layer.multihead_attn(
                output,
                embeddings,
                embeddings,
                need_weights=True,
                average_attn_weights=False,
            )
            output = output + dec_layer.dropout2(tgt2)
            output = dec_layer.norm2(output)
            # Feedforward block
            tgt2 = dec_layer.linear2(dec_layer.dropout(dec_layer.activation(dec_layer.linear1(output))))
            output = output + dec_layer.dropout3(tgt2)
            output = dec_layer.norm3(output)
            # Use CLS token only, keep heads separate.
            decoder_attn = attn_weights[:, :, 0, :].detach().cpu()  # (B, heads, src_len)
            for b in range(decoder_attn.size(0)):
                patch_images = [patches[b, i].detach().cpu() for i in range(decoder_attn.size(2))]
                head_labels = [f"h{i}" for i in range(decoder_attn.size(1))]
                # plot patches on Y, heads on X
                # patches on X axis, heads on Y axis
                agg = decoder_attn[b].mean(dim=0)
                agg = agg / agg.sum().clamp(min=1e-8)
                _save_heatmap(
                    decoder_attn[b],
                    os.path.join(out_dir, f"decoder_cross_attention_b{b:02d}.png"),
                    title=f"Decoder cross-attention (last layer, CLS only) | batch {b}",
                    xlabel="Patch index",
                    ylabel="Attention head",
                    x_patches=patch_images,
                    x_strip=agg,
                    normalize=False,
                    annotate=annotate,
                )
            # Token cosine similarity (CLS + queries)
            decoded = model.decoder(embeddings)[1].detach().cpu()  # (B, tokens, D)
            tokens = decoded
            tokens = torch.nn.functional.normalize(tokens, dim=2)
            for b in range(tokens.size(0)):
                sim = tokens[b] @ tokens[b].transpose(0, 1)
                token_labels = ["cls"] + [f"q{i}" for i in range(tokens.size(1) - 1)]
                _save_heatmap(
                    sim,
                    os.path.join(out_dir, f"decoder_token_cosine_b{b:02d}.png"),
                    title=f"Decoder token cosine | batch {b}",
                    xlabel="Token index",
                    ylabel="Token index",
                    x_labels=token_labels,
                    normalize=True,
                    annotate=annotate,
                )

        # Patch cosine similarity (encoder output) with patch thumbnails.
        if embeddings is not None:
            enc_norm = torch.nn.functional.normalize(embeddings, dim=2).detach().cpu()
            for b in range(enc_norm.size(0)):
                sim = enc_norm[b] @ enc_norm[b].transpose(0, 1)
                patch_images = [patches[b, i].detach().cpu() for i in range(enc_norm.size(1))]
                _save_heatmap(
                    sim,
                    os.path.join(out_dir, f"encoder_patch_cosine_b{b:02d}.png"),
                    title=f"Encoder patch cosine | batch {b}",
                    xlabel="Patch index",
                    ylabel="Patch index",
                    x_patches=patch_images,
                    y_patches=patch_images,
                    normalize=True,
                    annotate=annotate,
                )

        # Patch cosine similarity (pre-encoder, after patch_proj) with patch thumbnails.
        proj_norm = torch.nn.functional.normalize(proj_first, dim=2)
        for b in range(proj_norm.size(0)):
            sim = proj_norm[b] @ proj_norm[b].transpose(0, 1)
            patch_images = [patches[b, i].detach().cpu() for i in range(proj_norm.size(1))]
            _save_heatmap(
                sim,
                os.path.join(out_dir, f"pre_encoder_patch_cosine_b{b:02d}.png"),
                title=f"Pre-encoder patch cosine | batch {b}",
                xlabel="Patch index",
                ylabel="Patch index",
                x_patches=patch_images,
                y_patches=patch_images,
                normalize=True,
                annotate=annotate,
            )

        # Combined cosine (lower-left pre-encoder, upper-right encoder) + aggregate attention strip.
        if embeddings is not None and decoder_attn is not None:
            for b in range(proj_norm.size(0)):
                pre_cos = proj_norm[b] @ proj_norm[b].transpose(0, 1)
                enc_cos = enc_norm[b] @ enc_norm[b].transpose(0, 1)
                combined = torch.tril(pre_cos, diagonal=-1) + torch.triu(enc_cos, diagonal=0)
                agg = decoder_attn[b].mean(dim=0)
                agg = agg / agg.sum().clamp(min=1e-8)
                patch_images = [patches[b, i].detach().cpu() for i in range(combined.size(0))]
                _save_heatmap(
                    combined,
                    os.path.join(out_dir, f"combined_cosine_attention_b{b:02d}.png"),
                    title=f"Combined cosine (pre/enc) + aggregate attention | batch {b}",
                    xlabel="Patch index",
                    ylabel="Patch index",
                    x_patches=patch_images,
                    y_patches=patch_images,
                    x_strip=agg,
                    normalize=True,
                    annotate=annotate,
                    cmap="gray",
                )

    if encoder_attn is None:
        print("Visualization: no encoder attention available (no patch_encoder layers).")
    if decoder_attn is None:
        print("Visualization: no decoder cross-attention available (no decoder layers).")
    #x_scale = patch_px if (show_x or x_labels is not None or show_hist) else 1
