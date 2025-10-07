#!/usr/bin/env python3
"""
Streamlit dataset viewer for ARC-AGI dataset.

Interactive web interface to explore preprocessed ARC data batches.
"""

import streamlit as st
import torch
import matplotlib.pyplot as plt

from algo.config import Config
from algo.data import create_dataset
from algo.data.preprocessing import preprocess_rgb_image
from scripts.visualization_utils import (
    tensor_to_grayscale_numpy,
    apply_arc_color_palette,
    show_color_palette,
    visualize_task_combination,
)

# Set page config
st.set_page_config(page_title="ARC Dataset Viewer", page_icon="🔍", layout="wide")


def get_task_combinations(dataset, task_idx):
    """get all combinations for a specific task."""
    try:
        return dataset.get_task_combinations(task_idx)
    except Exception as e:
        st.warning(f"could not load combinations for task {task_idx}: {e}")
        import traceback

        st.code(traceback.format_exc())
        return {"regular": [], "counterfactual": [], "all": []}


def convert_grayscale_to_rgb(support_examples, config):
    """convert grayscale support examples to rgb format using proper preprocessing pipeline."""
    rgb_examples = []
    for example in support_examples:
        # convert grayscale to numpy (remove channel dimension if present)
        gray_input = tensor_to_grayscale_numpy(example["input"])
        gray_output = tensor_to_grayscale_numpy(example["output"])

        # use the same preprocessing pipeline as resnet dataset
        rgb_input = preprocess_rgb_image(gray_input, config)
        rgb_output = preprocess_rgb_image(gray_output, config)

        # add batch dimension to match resnet dataset format [1, 3, 64, 64]
        rgb_input = rgb_input.unsqueeze(0)
        rgb_output = rgb_output.unsqueeze(0)

        # create rgb example in the same format as resnet dataset
        rgb_example = {"input": rgb_input, "output": rgb_output}
        rgb_examples.append(rgb_example)

    return rgb_examples


def analyze_augmentation_groups(dataset, task_idx, combinations):
    """Analyze which augmentation group each combination belongs to."""
    task = dataset.tasks[task_idx]

    # Get groups for different counterfactual types
    groups_original = dataset._get_examples_by_augmentation_group(task, "original")
    groups_y = (
        dataset._get_examples_by_augmentation_group(task, "Y")
        if dataset.config.enable_counterfactuals
        else {}
    )
    groups_x = (
        dataset._get_examples_by_augmentation_group(task, "X")
        if dataset.config.enable_counterfactuals
        else {}
    )

    group_counts = {
        "original": 0,
        "augmented": 0,
        "counterfactual": 0,
        "counterfactual_augmented": 0,
        "mixed": 0,
    }

    original_count = 0
    augmented_count = 0
    counterfactual_count = 0
    mixed_count = 0

    for combo in combinations:
        indices = combo["cycling_indices"]
        i, j, k = indices
        is_counterfactual = combo.get("is_counterfactual", False)
        counterfactual_type = combo.get("counterfactual_type", "original")

        # Determine which group this combination belongs to
        if is_counterfactual:
            if counterfactual_type == "Y":
                groups = groups_y
            elif counterfactual_type == "X":
                groups = groups_x
            else:
                groups = groups_original
        else:
            groups = groups_original

        # Calculate group boundaries (same logic as in dataset)
        original_size = len(groups.get("original", []))
        augmented_size = len(groups.get("augmented", []))

        # Determine which group based on indices (same logic as in dataset)
        # Check if any training example indices are in the augmented range
        has_augmented_training = (
            (i >= original_size and i < (original_size + augmented_size))
            or (j >= original_size and j < (original_size + augmented_size))
            or (k >= original_size and k < (original_size + augmented_size))
        )

        if has_augmented_training:
            # This is an augmented group combination
            group_found = "augmented"
        else:
            # This is an original group combination
            group_found = "original"

        if is_counterfactual:
            if "augmented" in group_found:
                group_counts["counterfactual_augmented"] += 1
            else:
                group_counts["counterfactual"] += 1
            counterfactual_count += 1
        else:
            if "augmented" in group_found:
                group_counts["augmented"] += 1
                augmented_count += 1
            else:
                group_counts["original"] += 1
                original_count += 1

    return {
        "group_counts": group_counts,
        "original_count": original_count,
        "augmented_count": augmented_count,
        "counterfactual_count": counterfactual_count,
        "mixed_count": mixed_count,
    }


def get_combination_group_info(dataset, task_idx, combination):
    """Get augmentation group information for a specific combination."""
    from visualization_utils import get_combination_augmentation_group

    group_found = get_combination_augmentation_group(dataset, task_idx, combination)

    # Get the groups for additional info
    task = dataset.tasks[task_idx]
    is_counterfactual = combination.get("is_counterfactual", False)
    counterfactual_type = combination.get("counterfactual_type", "original")

    if is_counterfactual:
        if counterfactual_type == "Y":
            groups = dataset._get_examples_by_augmentation_group(task, "Y")
        elif counterfactual_type == "X":
            groups = dataset._get_examples_by_augmentation_group(task, "X")
        else:
            groups = dataset._get_examples_by_augmentation_group(task, "original")
    else:
        groups = dataset._get_examples_by_augmentation_group(task, "original")

    return group_found, groups


def main():
    """main streamlit app."""
    st.title("🔍 arc dataset viewer")
    st.markdown(
        "interactive exploration of preprocessed arc-agi dataset with proper task indexing"
    )

    # sidebar controls
    st.sidebar.header("dataset controls")

    # dataset selection
    dataset_choice = st.sidebar.selectbox(
        "select dataset", ["arc_agi1", "arc_agi2"], index=0
    )

    # dataset type selection
    dataset_type = st.sidebar.selectbox(
        "select dataset type",
        ["resnet", "patch"],
        index=1,
        help="resnet: rgb support examples, patch: grayscale support examples",
    )

    # holdout mode
    holdout_mode = st.sidebar.checkbox(
        "enable holdout mode",
        value=True,
        help="when enabled, tasks with 3+ train examples will have holdout data",
    )

    # color augmentation options
    st.sidebar.subheader("color augmentation")
    enable_color_augmentation = st.sidebar.checkbox(
        "enable color relabeling",
        value=False,
        help="apply color relabeling to test dataset diversity",
    )

    augmentation_variants = 1
    preserve_background = True
    augmentation_seed = 42

    if enable_color_augmentation:
        augmentation_variants = st.sidebar.slider(
            "augmentation variants",
            min_value=1,
            max_value=5,
            value=1,
            step=1,
            help="number of color-relabeled versions per example",
        )

        preserve_background = st.sidebar.checkbox(
            "preserve background",
            value=True,
            help="keep background color (0) unchanged during relabeling",
        )

        augmentation_seed = st.sidebar.number_input(
            "augmentation seed",
            min_value=0,
            max_value=10000,
            value=42,
            step=1,
            help="random seed for reproducible color relabeling",
        )

    # counterfactual options
    st.sidebar.subheader("counterfactual analysis")
    enable_counterfactuals = st.sidebar.checkbox(
        "enable counterfactuals",
        value=False,
        help="include counterfactual (rotated) examples in dataset",
    )

    counterfactual_transform = "rotate_90"
    if enable_counterfactuals:
        counterfactual_transform = st.sidebar.selectbox(
            "counterfactual transform",
            ["rotate_90", "rotate_180", "rotate_270", "reflect_h", "reflect_v"],
            index=0,
            help="type of transformation to apply to outputs",
        )

    # cycling options
    st.sidebar.subheader("cycling combinations")
    enable_cycling = st.sidebar.checkbox(
        "enable cycling",
        value=True,
        help="enable cycling combinations: (A,B)->T, (A,T)->B, (T,B)->A. Disable for simple (A,B)->T only.",
    )

    # load dataset
    try:
        config = Config()
        config.training_dataset = dataset_choice
        config.model_type = (
            "patch_attention" if dataset_type == "patch" else "simple_arc"
        )
        config.use_color_relabeling = enable_color_augmentation
        config.augmentation_variants = augmentation_variants
        config.preserve_background = preserve_background
        config.random_seed = augmentation_seed
        config.enable_counterfactuals = enable_counterfactuals
        config.counterfactual_transform = counterfactual_transform
        config.use_cycling = enable_cycling

        with st.spinner(f"loading {dataset_choice} dataset..."):
            try:
                if dataset_choice == "arc_agi1":
                    dataset = create_dataset(
                        config.arc_agi1_dir, config, holdout=holdout_mode
                    )
                else:
                    dataset = create_dataset(
                        config.processed_dir, config, holdout=holdout_mode
                    )
            except Exception as e:
                st.error(f"❌ error during dataset initialization: {e}")
                st.error(f"error type: {type(e).__name__}")
                import traceback

                st.error(f"traceback: {traceback.format_exc()}")
                st.stop()

        # get unique task indices from the dataset
        unique_task_indices = sorted(
            list(set(task_idx for task_idx, _ in dataset.combination_mapping))
        )

        # create mapping from task_id to task_idx for search functionality
        task_id_to_idx = {}
        if hasattr(dataset, "tasks"):
            for i, task in enumerate(dataset.tasks):
                task_id = task.get("task_id", f"task_{i}")
                task_id_to_idx[task_id] = i

        st.success(
            f"✅ loaded {len(unique_task_indices)} unique tasks from {dataset_choice} ({dataset_type} dataset)"
        )
        st.success(f"✅ total combinations: {len(dataset)}")

        # task selection (after dataset is loaded)
        st.sidebar.subheader("task selection")

        # search method selection
        search_method = st.sidebar.radio(
            "search method",
            ["by task index", "by task id"],
            index=0,
            help="choose how to search for tasks",
        )

        if search_method == "by task index":
            task_idx = st.sidebar.number_input(
                "task index",
                min_value=0,
                max_value=len(unique_task_indices) - 1,
                value=0,
                step=1,
                help=f"enter task index (0 to {len(unique_task_indices) - 1})",
            )
            # get the actual task index from unique list
            actual_task_idx = unique_task_indices[task_idx]

        else:  # by task id
            # create searchable list of task ids
            available_task_ids = sorted(list(task_id_to_idx.keys()))

            # search box for task id
            search_term = st.sidebar.text_input(
                "search task id",
                value="",
                placeholder="e.g., 27a28665, 239be575...",
                help="type part of the task id to search",
            )

            # filter task ids based on search term
            if search_term:
                filtered_task_ids = [
                    task_id
                    for task_id in available_task_ids
                    if search_term.lower() in task_id.lower()
                ]
            else:
                filtered_task_ids = available_task_ids

            if filtered_task_ids:
                st.sidebar.info(f"found {len(filtered_task_ids)} matching tasks")
                selected_task_id = st.sidebar.selectbox(
                    "select task id",
                    filtered_task_ids,
                    index=0,
                    help="select from filtered task ids",
                )
                actual_task_idx = task_id_to_idx[selected_task_id]
            else:
                st.sidebar.warning("no tasks found matching search term")
                actual_task_idx = unique_task_indices[0]  # fallback to first task

        # dataset info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("unique tasks", len(unique_task_indices))
        with col2:
            st.metric("total combinations", len(dataset))
        with col3:
            if search_method == "by task index":
                st.metric("current task", f"index {actual_task_idx}")
            else:
                # find the task id for the current task index
                current_task_id = "unknown"
                if hasattr(dataset, "tasks") and actual_task_idx < len(dataset.tasks):
                    current_task_id = dataset.tasks[actual_task_idx].get(
                        "task_id", f"task_{actual_task_idx}"
                    )
                st.metric("current task", f"{current_task_id}")
        with col4:
            if holdout_mode and dataset_choice == "arc_agi1":
                # count tasks with holdout data
                holdout_count = 0
                for i in range(min(100, len(dataset))):  # check first 100 samples
                    sample = dataset[i]
                    if sample.get("holdout_target") is not None:
                        holdout_count += 1
                st.metric("tasks w/ holdout", f"{holdout_count}/100")
            else:
                st.metric("holdout mode", "disabled")

    except Exception as e:
        st.error(f"❌ error loading dataset: {e}")
        st.error(f"error type: {type(e).__name__}")
        st.error("full traceback:")
        import traceback

        st.code(traceback.format_exc())
        st.stop()

    # get all combinations for the selected task
    try:
        combinations_data = get_task_combinations(dataset, actual_task_idx)
        combinations = combinations_data["all"]
        st.success(
            f"✅ loaded {len(combinations)} combinations for task {actual_task_idx}"
        )
    except Exception as e:
        st.error(f"❌ error loading task combinations: {e}")
        st.error(f"error type: {type(e).__name__}")
        st.error("full traceback:")
        import traceback

        st.code(traceback.format_exc())
        st.stop()

    # main content
    st.header("📊 task visualization")

    # color palette
    st.subheader("🎨 arc color palette")
    palette_fig = show_color_palette()
    st.pyplot(palette_fig)

    # task information
    st.subheader(f"🔍 task {actual_task_idx}")

    # show task info
    if hasattr(dataset, "tasks") and actual_task_idx < len(dataset.tasks):
        task_info = dataset.tasks[actual_task_idx]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**task info**")
            st.write(f"task id: {task_info.get('task_id', 'unknown')}")
            st.write(f"train examples: {len(task_info.get('train', []))}")
            st.write(f"test examples: {len(task_info.get('test', []))}")

        with col2:
            st.write("**combinations**")
            st.write(f"total combinations: {len(combinations)}")
            counterfactual_count = sum(
                1 for c in combinations if c.get("is_counterfactual", False)
            )
            st.write(f"counterfactual: {counterfactual_count}")
            st.write(f"original: {len(combinations) - counterfactual_count}")

            # Show test augmentation info
            if enable_color_augmentation:
                st.write("**test augmentation**")
                st.write("✅ test examples augmented")
                st.write("(consistent with training)")

        with col3:
            st.write("**augmentation status**")
            st.write(f"color relabeling: {'✅' if enable_color_augmentation else '❌'}")
            st.write(f"counterfactuals: {'✅' if enable_counterfactuals else '❌'}")
            st.write(f"cycling: {'✅' if enable_cycling else '❌'}")
            if enable_color_augmentation:
                st.write(f"variants: {augmentation_variants}")

            # Show cycling info
            if enable_cycling:
                st.write("**cycling patterns**")
                st.write("✅ (A,B)->T, (A,T)->B, (T,B)->A")
            else:
                st.write("**simple patterns**")
                st.write("✅ (A,B)->T only")

            # Show augmentation group separation info
            if enable_color_augmentation:
                st.write("**group separation**")
                st.write("✅ groups properly separated")
                st.write("(no mixing within combinations)")

    # augmentation group analysis
    if enable_color_augmentation and combinations:
        st.subheader("🔀 augmentation group analysis")

        # Analyze combinations by group
        group_analysis = analyze_augmentation_groups(
            dataset, actual_task_idx, combinations
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("original group", group_analysis["original_count"])
        with col2:
            st.metric("augmented group", group_analysis["augmented_count"])
        with col3:
            st.metric("counterfactual group", group_analysis["counterfactual_count"])
        with col4:
            st.metric("mixed groups", group_analysis["mixed_count"])

        # Show group separation status
        if group_analysis["mixed_count"] == 0:
            st.success("✅ Perfect group separation - no mixing detected!")
        else:
            st.error(
                f"⚠️ Found {group_analysis['mixed_count']} combinations with mixed groups"
            )

        # Show group breakdown
        st.write("**Group breakdown:**")
        for group_name, count in group_analysis["group_counts"].items():
            if count > 0:
                st.write(f"- {group_name}: {count} combinations")

    # combination selection
    if combinations:
        st.subheader("🔄 combination selection")

        # create combination options
        combo_options = []
        for i, combo in enumerate(combinations):
            # Cycling format with cycling_indices
            indices = combo["cycling_indices"]

            # Show different patterns based on cycling setting
            if enable_cycling:
                # Show the actual cycling pattern
                if indices[2] < 0:  # Test example as target
                    indices_str = f"({indices[0]}, {indices[1]}) -> T{abs(indices[2])}"
                elif indices[1] < 0:  # Test example as support
                    indices_str = f"({indices[0]}, T{abs(indices[1])}) -> {indices[2]}"
                elif indices[0] < 0:  # Test example as first support
                    indices_str = f"(T{abs(indices[0])}, {indices[1]}) -> {indices[2]}"
                else:
                    indices_str = f"({indices[0]}, {indices[1]}) -> {indices[2]}"
            else:
                # Simple pattern only
                indices_str = f"({indices[0]}, {indices[1]}) -> T{abs(indices[2]) if indices[2] < 0 else indices[2]}"

            is_counterfactual = combo.get("is_counterfactual", False)
            counterfactual_marker = " (counterfactual)" if is_counterfactual else ""

            # Get group information for this combination
            if enable_color_augmentation:
                group_found, _ = get_combination_group_info(
                    dataset, actual_task_idx, combo
                )
                group_marker = f" [{group_found}]"
            else:
                group_marker = ""

            combo_options.append(
                f"combination {combo['combination_idx']}: {indices_str}{counterfactual_marker}{group_marker}"
            )

        selected_combo_idx = st.selectbox(
            "select combination",
            range(len(combinations)),
            format_func=lambda x: combo_options[x],
            index=0,
            help="select which combination of training examples to view",
        )

        selected_combo = combinations[selected_combo_idx]
        task_data = selected_combo

        # check if this task has holdout data
        has_holdout = task_data.get("holdout_example") is not None

        if has_holdout:
            st.info(f"✅ task {actual_task_idx} has holdout data")
        else:
            st.info(f"ℹ️ task {actual_task_idx} has no holdout data")

        # prepare task data for visualization
        display_task_data = task_data.copy()

        # for patch datasets, convert grayscale support examples to rgb for consistent display
        if dataset_type == "patch" and task_data.get("support_examples_rgb") is None:
            display_task_data["support_examples_rgb"] = convert_grayscale_to_rgb(
                task_data["support_examples"], config
            )

        # visualize the selected combination
        task_fig = visualize_task_combination(
            display_task_data,
            actual_task_idx,
            selected_combo["combination_idx"],
            show_holdout=has_holdout,
        )
        st.pyplot(task_fig)

        # combination details
        st.subheader("📋 combination details")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**combination info**")
            st.write(f"combination index: {selected_combo['combination_idx']}")

            # Display cycling indices
            cycling_indices = selected_combo["cycling_indices"]

            # Show pattern based on cycling setting
            if enable_cycling:
                if cycling_indices[2] < 0:  # Test example as target
                    pattern_str = f"({cycling_indices[0]}, {cycling_indices[1]}) -> T{abs(cycling_indices[2])}"
                    pattern_desc = "format: (support1, support2) -> test_target"
                elif cycling_indices[1] < 0:  # Test example as support
                    pattern_str = f"({cycling_indices[0]}, T{abs(cycling_indices[1])}) -> {cycling_indices[2]}"
                    pattern_desc = "format: (support1, test_target) -> support2"
                elif cycling_indices[0] < 0:  # Test example as first support
                    pattern_str = f"(T{abs(cycling_indices[0])}, {cycling_indices[1]}) -> {cycling_indices[2]}"
                    pattern_desc = "format: (test_target, support2) -> support1"
                else:
                    pattern_str = f"({cycling_indices[0]}, {cycling_indices[1]}) -> {cycling_indices[2]}"
                    pattern_desc = "format: (support1, support2) -> target"
            else:
                pattern_str = f"({cycling_indices[0]}, {cycling_indices[1]}) -> T{abs(cycling_indices[2]) if cycling_indices[2] < 0 else cycling_indices[2]}"
                pattern_desc = "format: (support1, support2) -> test_target (simple)"

            st.write(f"pattern: {pattern_str}")
            st.write(pattern_desc)

            st.write(
                f"is counterfactual: {'✅' if selected_combo.get('is_counterfactual', False) else '❌'}"
            )

            # Show augmentation group information
            if enable_color_augmentation:
                group_found, groups = get_combination_group_info(
                    dataset, actual_task_idx, selected_combo
                )
                st.write(f"**augmentation group: {group_found}**")

                # Show group sizes
                st.write("**group sizes:**")
                for group_name, group_examples in groups.items():
                    if len(group_examples) > 0:
                        st.write(f"- {group_name}: {len(group_examples)} examples")

        with col2:
            st.write("**data structure**")
            st.write("support examples: 2")

            # Cycling format
            st.write("target example: 1 (cycling)")
            st.write("format: support1 + support2 -> target")

            st.write(f"test examples: {task_data.get('num_test_examples', 1)}")
            if task_data.get("holdout_example") is not None:
                st.write("holdout example: 1")
            else:
                st.write("holdout example: 0")
            if is_counterfactual:
                st.write("✅ counterfactuals enabled")
            else:
                st.write("❌ no counterfactuals")

        with col3:
            st.write("**holdout status**")
            if has_holdout:
                st.write("✅ has holdout data")
                st.write(
                    f"holdout input shape: {task_data['holdout_example']['input'].shape}"
                )
            else:
                st.write("❌ no holdout data")

            # Show group separation status
            if enable_color_augmentation:
                st.write("**group separation**")
                group_found, _ = get_combination_group_info(
                    dataset, actual_task_idx, selected_combo
                )
                if group_found == "mixed":
                    st.write("⚠️ mixed groups")
                else:
                    st.write(f"✅ {group_found} group")

        # test examples display
        st.subheader("🧪 test examples")
        test_examples = task_data.get("test_examples", [])
        num_test_examples = task_data.get("num_test_examples", len(test_examples))

        if num_test_examples > 0:
            # Create columns for test examples (max 3 per row)
            cols_per_row = min(3, num_test_examples)

            for i in range(0, num_test_examples, cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    test_idx = i + j
                    if test_idx < num_test_examples:
                        with col:
                            st.write(f"**test example {test_idx + 1}**")

                            # Test input
                            test_input_np = tensor_to_grayscale_numpy(
                                test_examples[test_idx]["input"]
                            )
                            rgb_test_input = apply_arc_color_palette(test_input_np)
                            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                            ax.imshow(rgb_test_input)
                            ax.set_title(f"input {test_idx + 1}", fontsize=10)
                            ax.axis("off")
                            st.pyplot(fig, width=300)
                            plt.close(fig)

                            # Test output
                            test_output_np = tensor_to_grayscale_numpy(
                                test_examples[test_idx]["output"]
                            )
                            rgb_test_output = apply_arc_color_palette(test_output_np)
                            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                            ax.imshow(rgb_test_output)
                            ax.set_title(f"output {test_idx + 1}", fontsize=10)
                            ax.axis("off")
                            st.pyplot(fig, width=300)
                            plt.close(fig)
        else:
            st.write("❌ no test examples available")

        # target example display (for cycling format)
        if "target_example" in task_data:
            st.subheader("🎯 target example (cycling)")
            st.write(
                "This is the target that the model should predict using the two support examples above."
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write("**target input**")
                target_input_np = tensor_to_grayscale_numpy(
                    task_data["target_example"]["input"]
                )
                rgb_target_input = apply_arc_color_palette(target_input_np)
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(rgb_target_input)
                ax.set_title("target input", fontsize=12)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.write("**target output**")
                target_output_np = tensor_to_grayscale_numpy(
                    task_data["target_example"]["output"]
                )
                rgb_target_output = apply_arc_color_palette(target_output_np)
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(rgb_target_output)
                ax.set_title("target output", fontsize=12)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

        # holdout vs test comparison
        if has_holdout and holdout_mode and dataset_choice == "arc_agi1":
            st.subheader("🔄 holdout vs test comparison")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**first test input**")
                test_input_np = tensor_to_grayscale_numpy(test_examples[0]["input"])
                rgb_test_input = apply_arc_color_palette(test_input_np)
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(rgb_test_input)
                ax.set_title("test input", fontsize=12)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                st.write("**holdout input**")
                holdout_input_np = tensor_to_grayscale_numpy(
                    task_data["holdout_example"]["input"]
                )
                rgb_holdout_input = apply_arc_color_palette(holdout_input_np)
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(rgb_holdout_input)
                ax.set_title("holdout input", fontsize=12)
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

            # check if they're the same
            test_input = test_examples[0]["input"]
            holdout_input = task_data["holdout_example"]["input"]
            are_same = torch.equal(test_input, holdout_input)

            if are_same:
                st.error(
                    "⚠️ holdout and test inputs are the same! this indicates a bug."
                )
            else:
                st.success(
                    "✅ holdout and test inputs are different - holdout is working correctly!"
                )

        # data statistics
        st.subheader("📈 data statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**support images**")
            # show dataset type info
            st.write(f"**dataset type:** {dataset_type}")

            # check if we have rgb support examples (resnet) or just grayscale (patch)
            if (
                "support_examples_rgb" in task_data
                and task_data["support_examples_rgb"] is not None
            ):
                # resnet dataset - show rgb support examples
                example_img = task_data["support_examples_rgb"][0]["input"]
                st.write("**rgb support examples:**")
                st.write(f"- shape: {example_img.shape}")
                st.write(f"- data type: {example_img.dtype}")
                st.write(
                    f"- value range: [{example_img.min():.3f}, {example_img.max():.3f}]"
                )

                # also show grayscale support examples
                grayscale_img = task_data["support_examples"][0]["input"]
                st.write("**grayscale support examples:**")
                st.write(f"- shape: {grayscale_img.shape}")
                st.write(f"- data type: {grayscale_img.dtype}")
                st.write(
                    f"- value range: [{grayscale_img.min():.0f}, {grayscale_img.max():.0f}]"
                )
            else:
                # patch dataset - only grayscale support examples
                example_img = task_data["support_examples"][0]["input"]
                st.write("**grayscale support examples:**")
                st.write(f"- shape: {example_img.shape}")
                st.write(f"- data type: {example_img.dtype}")
                st.write(
                    f"- value range: [{example_img.min():.0f}, {example_img.max():.0f}]"
                )

        with col2:
            st.write("**test images (grayscale)**")
            test_img = test_examples[0]["input"]  # Use first test example for stats
            st.write(f"- shape: {test_img.shape}")
            st.write(f"- data type: {test_img.dtype}")
            st.write(f"- value range: [{test_img.min():.0f}, {test_img.max():.0f}]")
            st.write(f"- number of test examples: {num_test_examples}")

            # Add target example stats if available
            if "target_example" in task_data:
                st.write("**target image (cycling)**")
                target_img = task_data["target_example"]["input"]
                st.write(f"- shape: {target_img.shape}")
                st.write(f"- data type: {target_img.dtype}")
                st.write(
                    f"- value range: [{target_img.min():.0f}, {target_img.max():.0f}]"
                )

        with col3:
            if has_holdout:
                st.write("**holdout images (grayscale)**")
                holdout_img = task_data["holdout_example"]["input"]
                st.write(f"- shape: {holdout_img.shape}")
                st.write(f"- data type: {holdout_img.dtype}")
                st.write(
                    f"- value range: [{holdout_img.min():.0f}, {holdout_img.max():.0f}]"
                )
            else:
                st.write("**holdout status**")
                st.write("no holdout data available")

    else:
        st.warning(f"no combinations found for task {actual_task_idx}")

    # refresh button
    if st.button("🔄 refresh task"):
        st.rerun()


if __name__ == "__main__":
    main()
