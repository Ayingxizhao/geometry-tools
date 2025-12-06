#!/usr/bin/env python3
"""
Agent Benchmark V2 with Self-Reflection

Model: GPT-5 (fixed for both conditions)

This benchmark tests two agent conditions:
  A. LLM-only reflection: Agent (GPT-5, no image) + self-reflection (GPT-5, no image)
  B. VLM-guided reflection: Agent (GPT-5, no image) + VLM evaluator (GPT-5 with image)

Key changes from v1:
1. Uses ONLY new geometry_tools APIs (no cv2/OpenCV)
2. Self-reflection up to 3 rounds (not just on error, but on uncertainty)
3. Two experimental conditions for comparison
4. Agent never sees the image directly - only API results
5. In condition B, GPT-5 with image input serves as evaluator

Usage:
    # Condition A: LLM-only reflection (no image anywhere)
    python benchmark_agent_v2.py --condition A --seed 42

    # Condition B: VLM-guided reflection (evaluator sees image)
    python benchmark_agent_v2.py --condition B --seed 42
"""

import argparse
import base64
import json
import logging
import os
import random
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from io import StringIO
from pathlib import Path
from threading import Lock
from typing import List, Literal, Optional

# Global logger
benchmark_logger = None


def setup_logging(log_file: str = None) -> logging.Logger:
    """Setup logging to file and console."""
    global benchmark_logger

    logger = logging.getLogger("benchmark_agent_v2")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (DEBUG level)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    benchmark_logger = logger
    return logger


def create_tracked_function(func, func_name: str, call_tracker: list):
    """Create a wrapper that tracks function calls."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Don't include image data in logs
        logged_args = []
        for a in args:
            if isinstance(a, str) and len(a) < 200:
                logged_args.append(a)
            else:
                logged_args.append(f"<{type(a).__name__}>")

        call_info = {
            "function": func_name,
            "args": logged_args,
            "kwargs": {k: str(v)[:100] for k, v in kwargs.items()},
        }

        result = func(*args, **kwargs)
        call_info["result_success"] = (
            result.get("success", None) if isinstance(result, dict) else None
        )
        call_tracker.append(call_info)

        if benchmark_logger:
            benchmark_logger.debug(
                f"    API CALL: {func_name}() -> success={call_info['result_success']}"
            )

        return result

    return wrapper


@dataclass
class BenchmarkSample:
    """A single benchmark sample."""

    image_path: str
    task_type: str
    subtask: str
    ground_truth: bool
    question: str


@dataclass
class ReflectionRound:
    """A single reflection round."""

    round_num: int
    code: str
    reasoning: str
    api_result: dict
    answer: Optional[str]
    confidence: str  # "high", "medium", "low", "uncertain"
    reflection_feedback: str  # Feedback from reflector (LLM or VLM)
    needs_retry: bool


@dataclass
class BenchmarkResult:
    """Result for a single sample."""

    sample: BenchmarkSample
    final_answer: str
    predicted: Optional[bool]
    correct: bool
    num_rounds: int
    reflection_rounds: list = field(default_factory=list)
    functions_called: list = field(default_factory=list)
    error: Optional[str] = None
    condition: str = ""  # "A" or "B"


# =============================================================================
# API Documentation for Agent (NEW APIs only)
# =============================================================================

API_DOCUMENTATION = """
## Available Geometry Tools API

You are an AI agent that analyzes geometry images using ONLY the provided API functions.
You do NOT have direct access to the image - you must use these APIs to extract information.

### Line Detection & Comparison APIs

1. `compare_all_lines(image_path)`
   Detect and compare all lines in the image.
   Returns: {
       'success': bool,
       'num_lines': int,
       'lines': list,           # List of line dicts SORTED BY LENGTH (longest first), each has 'id', 'length', 'midpoint'
       'longest_line': str,     # line_id of longest line
       'shortest_line': str     # line_id of shortest line
   }
   NOTE: Line IDs (e.g., "line_0", "line_1") are assigned by POSITION: line_0 = top line, line_1 = bottom line.
   But the 'lines' list is sorted by LENGTH, not by ID.

2. `is_line_longer(image_path, line_A_id, line_B_id, tolerance=0.05)`
   Compare two specific lines by ID.
   Returns: {
       'success': bool,
       'answer': bool,          # True if line_A > line_B
       'answer_text': str,      # "yes" or "no"
       'line_A_length': float,
       'line_B_length': float
   }

3. `measure_line_length(image_path, line_id)`
   Measure a specific line by ID.
   Returns: {
       'success': bool,
       'length': float,
       'line_id': str
   }

### Triangle Detection & Validation APIs

4. `detect_and_classify_triangles(image_path, min_area=1000.0, max_area=100000.0)`
   Detect all triangles and classify their types.
   Returns: {
       'success': bool,
       'num_triangles': int,
       'triangles': list,       # Each has 'id', 'classification' with 'type', 'angles', 'sides'
       'equilateral_triangles': list,  # IDs of equilateral triangles
       'right_triangles': list         # IDs of right triangles
   }

5. `is_valid_equilateral(image_path, triangle_id, angle_tolerance=5.0, side_ratio_tolerance=0.05)`
   Validate if a triangle is equilateral (all sides equal, all angles ~60 degrees).
   Returns: {
       'success': bool,
       'is_valid': bool,        # True if valid equilateral
       'confidence': str,       # "high", "medium", "low"
       'classification': dict   # Full triangle properties
   }

6. `is_valid_right_triangle(image_path, triangle_id, angle_tolerance=5.0)`
   Validate if a triangle is a right triangle (has a 90-degree angle).
   Returns: {
       'success': bool,
       'is_valid': bool,        # True if valid right triangle
       'confidence': str,
       'classification': dict
   }

7. `answer_triangle_question(image_path, question)`
   Natural language interface for triangle questions.
   Returns: {
       'success': bool,
       'is_valid': bool,
       'natural_answer': str,   # e.g., "Yes, this is a valid equilateral triangle."
       'confidence': str
   }

### Shape Comparison APIs (circles, squares, rectangles)

8. `compare_shapes_in_image(image_path, min_area=500.0, max_area=50000.0)`
   Detect shapes and compare their sizes (left shape vs right shape).
   Returns: {
       'success': bool,
       'num_shapes': int,
       'shapes': list,          # Detected shapes with 'type', 'area', 'centroid'
       'shape_1': dict,         # Left shape info
       'shape_2': dict,         # Right shape info
       'is_shape_1_larger': bool,  # True if left shape is larger
       'area_ratio': float,
       'area_difference': float
   }

9. `is_shape_larger(image_path, expected_larger=True)`
   Validate if left shape is larger than right shape.
   Returns: {
       'success': bool,
       'is_valid': bool,       # True if result matches expected_larger
       'shape_1_larger': bool,
       'area_ratio': float
   }

10. `answer_shape_comparison_question(image_path, question)`
    Natural language interface for shape comparison.
    Returns: {
        'success': bool,
        'is_shape_1_larger': bool,
        'natural_answer': str,  # e.g., "Yes, the left circle is larger..."
        'area_ratio': float
    }

### IMPORTANT RULES:
- DO NOT use any import statements! All functions are pre-loaded in the namespace.
- DO NOT write `from geometry_tools import ...` or `import geometry_tools` - this will cause errors!
- Just call the functions directly: `result = compare_all_lines(image_path)`
- You MUST use ONLY these API functions listed above
- You do NOT have access to cv2, OpenCV, or raw image data
- Your code must set `answer = "YES"` or `answer = "NO"`
- Your code must also set `confidence = "high"`, `"medium"`, `"low"`, or `"uncertain"`
- If confidence is not "high", you may get a chance to reflect and retry
"""


# =============================================================================
# Tool Definitions
# =============================================================================

CODE_EXEC_TOOL = {
    "type": "function",
    "name": "execute_geometry_api",
    "description": """Execute Python code using geometry_tools API to analyze the image.

IMPORTANT: DO NOT use any import statements! All functions are already available in the namespace.
DO NOT write: from geometry_tools import ... (this will cause errors!)
Just call functions directly: result = compare_shapes_in_image(image_path)

Your code has access to:
- `image_path`: Path to the image file (string)
- All geometry_tools API functions (already loaded, no import needed)
- `np` (NumPy) and `math` for calculations

Your code MUST set two variables:
1. `answer`: Either "YES" or "NO"
2. `confidence`: Either "high", "medium", "low", or "uncertain"

If you are uncertain, set confidence="uncertain" and you will get feedback to help.""",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code using geometry_tools API. Must set `answer` and `confidence` variables.",
            },
            "reasoning": {
                "type": "string",
                "description": "Your reasoning about what API to use and why.",
            },
        },
        "required": ["code", "reasoning"],
        "additionalProperties": False,
    },
    "strict": True,
}


# =============================================================================
# Code Execution Sandbox
# =============================================================================


def execute_code_sandbox(code: str, image_path: str) -> dict:
    """
    Execute Python code in a sandbox with geometry_tools APIs only.

    Returns:
        dict with 'success', 'answer', 'confidence', 'output', 'error', 'function_calls'
    """
    import importlib.util
    import math
    import traceback as tb_module  # Import at top to avoid reference errors
    from pathlib import Path

    import cv2
    import numpy as np

    function_call_tracker = []

    # Import geometry_tools APIs by directly loading submodules
    # (bypassing __init__.py which has missing dependencies)

    geometry_tools_available = {}
    try:
        # Get path to geometry_tools package
        pkg_dir = Path(__file__).parent / "geometry_tools"
        if not pkg_dir.exists():
            # Try relative to cwd
            pkg_dir = Path("geometry_tools")

        # Load submodules directly
        def load_module(name, filepath):
            spec = importlib.util.spec_from_file_location(name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        line_detection = load_module("line_detection", pkg_dir / "line_detection.py")
        triangle_detection = load_module(
            "triangle_detection", pkg_dir / "triangle_detection.py"
        )
        shape_detection = load_module("shape_detection", pkg_dir / "shape_detection.py")

        # Define preprocessing functions locally
        def load_image(image_path):
            """Load an image from file path."""
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            return img

        def preprocess_for_line_detection(image):
            """Preprocess image for line detection."""
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            return gray, edges

        # Define high-level API functions
        def compare_all_lines(image_path, filter_horizontal=True, tolerance=0.05):
            """Detect and compare all lines in the image."""
            image = (
                load_image(image_path) if isinstance(image_path, str) else image_path
            )
            gray, edges = preprocess_for_line_detection(image)
            lines = line_detection.detect_and_filter_lines(
                edges, filter_horizontal=filter_horizontal
            )
            if not lines:
                return {"success": False, "error": "No lines detected in image"}
            sorted_lines = sorted(lines, key=lambda x: x["length"], reverse=True)
            return {
                "success": True,
                "num_lines": len(lines),
                "lines": sorted_lines,
                "longest_line": sorted_lines[0]["id"],
                "shortest_line": sorted_lines[-1]["id"],
                "longest_length": sorted_lines[0]["length"],
                "shortest_length": sorted_lines[-1]["length"],
            }

        def is_line_longer(image_path, line_A_id, line_B_id, tolerance=0.05):
            """Compare two lines: Is line A longer than line B?"""
            image = (
                load_image(image_path) if isinstance(image_path, str) else image_path
            )
            gray, edges = preprocess_for_line_detection(image)
            lines = line_detection.detect_and_filter_lines(
                edges, filter_horizontal=True
            )
            line_A = line_B = None
            for line in lines:
                if line["id"] == line_A_id:
                    line_A = line
                if line["id"] == line_B_id:
                    line_B = line
            if line_A is None or line_B is None:
                return {
                    "success": False,
                    "error": f"Lines not found. Available: {[l['id'] for l in lines]}",
                }
            length_A, length_B = line_A["length"], line_B["length"]
            rel_diff = (
                abs(length_A - length_B) / max(length_A, length_B)
                if max(length_A, length_B) > 0
                else 0
            )
            are_equal = rel_diff < tolerance
            answer = length_A > length_B if not are_equal else False
            return {
                "success": True,
                "answer": answer,
                "answer_text": "yes" if answer else "no",
                "are_equal": are_equal,
                "line_A_length": length_A,
                "line_B_length": length_B,
            }

        def measure_line_length(image_path, line_id):
            """Measure a specific line by ID."""
            image = (
                load_image(image_path) if isinstance(image_path, str) else image_path
            )
            gray, edges = preprocess_for_line_detection(image)
            lines = line_detection.detect_and_filter_lines(
                edges, filter_horizontal=True
            )
            for line in lines:
                if line["id"] == line_id:
                    return {
                        "success": True,
                        "length": line["length"],
                        "line_id": line_id,
                    }
            return {
                "success": False,
                "error": f"Line '{line_id}' not found. Available: {[l['id'] for l in lines]}",
            }

        def detect_and_classify_triangles(
            image_path, min_area=1000.0, max_area=100000.0
        ):
            """Detect all triangles and classify their types."""
            image = (
                load_image(image_path) if isinstance(image_path, str) else image_path
            )
            gray, edges = preprocess_for_line_detection(image)
            triangles = triangle_detection.detect_and_classify_triangles(
                edges, min_area, max_area
            )
            if not triangles:
                return {"success": False, "error": "No triangles detected"}
            equilateral_ids = [
                t["id"]
                for t in triangles
                if t["classification"]["type"] == "equilateral"
            ]
            right_ids = [
                t["id"] for t in triangles if t["classification"]["type"] == "right"
            ]
            return {
                "success": True,
                "num_triangles": len(triangles),
                "triangles": triangles,
                "equilateral_triangles": equilateral_ids,
                "right_triangles": right_ids,
            }

        def is_valid_equilateral(
            image_path, triangle_id, angle_tolerance=5.0, side_ratio_tolerance=0.05
        ):
            """Check if a triangle is a valid equilateral."""
            image = (
                load_image(image_path) if isinstance(image_path, str) else image_path
            )
            gray, edges = preprocess_for_line_detection(image)
            triangles = triangle_detection.detect_and_classify_triangles(edges)
            target = next((t for t in triangles if t["id"] == triangle_id), None)
            if target is None:
                return {
                    "success": False,
                    "error": f"Triangle '{triangle_id}' not found. Available: {[t['id'] for t in triangles]}",
                }
            vertices = np.array(target["vertices"])
            validation = triangle_detection.validate_triangle(
                vertices, "equilateral", angle_tolerance, side_ratio_tolerance
            )
            return {
                "success": True,
                "is_valid": validation["is_valid"],
                "triangle_id": triangle_id,
                "confidence": validation["confidence"],
                "classification": target["classification"],
            }

        def is_valid_right_triangle(image_path, triangle_id, angle_tolerance=5.0):
            """Check if a triangle is a valid right triangle."""
            image = (
                load_image(image_path) if isinstance(image_path, str) else image_path
            )
            gray, edges = preprocess_for_line_detection(image)
            triangles = triangle_detection.detect_and_classify_triangles(edges)
            target = next((t for t in triangles if t["id"] == triangle_id), None)
            if target is None:
                return {
                    "success": False,
                    "error": f"Triangle '{triangle_id}' not found. Available: {[t['id'] for t in triangles]}",
                }
            vertices = np.array(target["vertices"])
            validation = triangle_detection.validate_triangle(
                vertices, "right", angle_tolerance
            )
            return {
                "success": True,
                "is_valid": validation["is_valid"],
                "triangle_id": triangle_id,
                "confidence": validation["confidence"],
                "classification": target["classification"],
            }

        def answer_triangle_question(image_path, question, triangle_id=None):
            """Natural language interface for triangle questions."""
            question_lower = question.lower()
            if "equilateral" in question_lower:
                expected_type = "equilateral"
            elif "right" in question_lower:
                expected_type = "right"
            else:
                return {
                    "success": False,
                    "error": "Could not determine triangle type from question",
                }
            if triangle_id is None:
                result = detect_and_classify_triangles(image_path)
                if not result["success"]:
                    return result
                triangle_id = (
                    result["triangles"][0]["id"]
                    if result["num_triangles"] >= 1
                    else None
                )
                if triangle_id is None:
                    return {"success": False, "error": "No triangles found"}
            result = (
                is_valid_equilateral(image_path, triangle_id)
                if expected_type == "equilateral"
                else is_valid_right_triangle(image_path, triangle_id)
            )
            if result.get("success"):
                result["natural_answer"] = (
                    f"Yes, this is a valid {expected_type} triangle."
                    if result["is_valid"]
                    else f"No, this is not a valid {expected_type} triangle."
                )
            return result

        def compare_shapes_in_image(image_path, min_area=500.0, max_area=50000.0):
            """Detect and compare sizes of shapes."""
            image = (
                load_image(image_path) if isinstance(image_path, str) else image_path
            )
            detection = shape_detection.detect_and_classify_shapes(
                image, min_area, max_area
            )
            if not detection["success"]:
                return {
                    "success": False,
                    "error": detection["error"],
                    "num_shapes": 0,
                    "shapes": [],
                }
            comparison = shape_detection.compare_shape_sizes(detection["shapes"])
            if not comparison["success"]:
                return {
                    "success": False,
                    "error": comparison["error"],
                    "num_shapes": detection["num_shapes"],
                    "shapes": detection["shapes"],
                }
            return {
                "success": True,
                "num_shapes": detection["num_shapes"],
                "shapes": detection["shapes"],
                "shape_1": comparison["shape_1"],
                "shape_2": comparison["shape_2"],
                "is_shape_1_larger": comparison["is_shape_1_larger"],
                "area_ratio": comparison["area_ratio"],
                "area_difference": comparison["area_difference"],
            }

        def is_shape_larger(
            image_path, expected_larger=True, min_area=500.0, max_area=50000.0
        ):
            """Validate if left shape is larger than right shape."""
            image = (
                load_image(image_path) if isinstance(image_path, str) else image_path
            )
            return shape_detection.validate_shape_comparison(
                image, expected_larger, min_area, max_area
            )

        def answer_shape_comparison_question(
            image_path, question, min_area=500.0, max_area=50000.0
        ):
            """Natural language interface for shape comparison."""
            result = compare_shapes_in_image(image_path, min_area, max_area)
            if not result["success"]:
                result["natural_answer"] = (
                    f"I couldn't analyze the shapes: {result.get('error', 'unknown error')}"
                )
                return result
            shape_1_larger = result["is_shape_1_larger"]
            s1_type, s2_type = result["shape_1"]["type"], result["shape_2"]["type"]
            ratio = result["area_ratio"]
            result["natural_answer"] = (
                f"Yes, the left {s1_type} is larger (ratio: {ratio:.2f})."
                if shape_1_larger
                else f"No, the left {s1_type} is not larger than the right {s2_type} (ratio: {ratio:.2f})."
            )
            return result

        # Wrap all functions with tracking
        geometry_tools_available = {
            "compare_all_lines": create_tracked_function(
                compare_all_lines, "compare_all_lines", function_call_tracker
            ),
            "is_line_longer": create_tracked_function(
                is_line_longer, "is_line_longer", function_call_tracker
            ),
            "measure_line_length": create_tracked_function(
                measure_line_length, "measure_line_length", function_call_tracker
            ),
            "detect_and_classify_triangles": create_tracked_function(
                detect_and_classify_triangles,
                "detect_and_classify_triangles",
                function_call_tracker,
            ),
            "is_valid_equilateral": create_tracked_function(
                is_valid_equilateral, "is_valid_equilateral", function_call_tracker
            ),
            "is_valid_right_triangle": create_tracked_function(
                is_valid_right_triangle,
                "is_valid_right_triangle",
                function_call_tracker,
            ),
            "answer_triangle_question": create_tracked_function(
                answer_triangle_question,
                "answer_triangle_question",
                function_call_tracker,
            ),
            "compare_shapes_in_image": create_tracked_function(
                compare_shapes_in_image,
                "compare_shapes_in_image",
                function_call_tracker,
            ),
            "is_shape_larger": create_tracked_function(
                is_shape_larger, "is_shape_larger", function_call_tracker
            ),
            "answer_shape_comparison_question": create_tracked_function(
                answer_shape_comparison_question,
                "answer_shape_comparison_question",
                function_call_tracker,
            ),
        }
    except Exception as e:
        error_msg = f"Failed to import geometry_tools: {e}\n{tb_module.format_exc()}"
        if benchmark_logger:
            benchmark_logger.error(f"Sandbox import error: {error_msg}")
        return {
            "success": False,
            "answer": None,
            "confidence": None,
            "output": "",
            "error": error_msg,
            "function_calls": [],
        }

    # Create a thread-local StringIO to capture print output (thread-safe)
    output_capture = StringIO()

    # Custom print function that writes to our local StringIO
    def local_print(*args, **kwargs):
        kwargs["file"] = output_capture
        print(*args, **kwargs)

    # Create restricted namespace (NO cv2, NO raw image)
    namespace = {
        "np": np,
        "numpy": np,
        "math": math,
        "image_path": image_path,
        "print": local_print,  # Override print to capture output
        **geometry_tools_available,
    }

    try:
        exec(code, namespace)
        output = output_capture.getvalue()

        answer = namespace.get("answer", None)
        confidence = namespace.get("confidence", "uncertain")

        if answer is None:
            return {
                "success": False,
                "answer": None,
                "confidence": confidence,
                "output": output,
                "error": "Code did not set 'answer' variable",
                "function_calls": function_call_tracker,
            }

        # Normalize answer
        answer_str = str(answer).strip().upper()
        if answer_str not in ["YES", "NO"]:
            return {
                "success": False,
                "answer": answer_str,
                "confidence": confidence,
                "output": output,
                "error": f"Answer must be 'YES' or 'NO', got: {answer_str}",
                "function_calls": function_call_tracker,
            }

        # Normalize confidence
        confidence_str = str(confidence).strip().lower()
        if confidence_str not in ["high", "medium", "low", "uncertain"]:
            confidence_str = "uncertain"

        return {
            "success": True,
            "answer": answer_str,
            "confidence": confidence_str,
            "output": output,
            "error": None,
            "function_calls": function_call_tracker,
        }

    except Exception as e:
        output = output_capture.getvalue()
        error_trace = tb_module.format_exc()
        error_msg = f"{str(e)}\n{error_trace}"
        if benchmark_logger:
            benchmark_logger.error(f"Sandbox execution error: {error_msg}")
        return {
            "success": False,
            "answer": None,
            "confidence": "uncertain",
            "output": output,
            "error": error_msg,
            "function_calls": function_call_tracker,
        }


# =============================================================================
# Reflection Logic
# =============================================================================


def get_llm_reflection(
    client,
    model: str,
    question: str,
    code: str,
    api_result: dict,
    previous_rounds: list,
) -> str:
    """
    Condition A: LLM-only reflection (no image access).

    The LLM reflects on the code and API results without seeing the image.
    """
    # Build context from previous rounds
    history = ""
    for r in previous_rounds:
        history += f"\n--- Round {r['round']} ---\n"
        history += f"Code: {r['code'][:500]}...\n"
        history += f"API Result: {json.dumps(r['api_result'], indent=2)[:500]}\n"
        history += f"Answer: {r['answer']}, Confidence: {r['confidence']}\n"

    prompt = f"""You are reviewing an AI agent's attempt to answer a geometry question.

Question: {question}

The agent wrote this code:
```python
{code}
```

API execution result:
{json.dumps(api_result, indent=2)}

{f"Previous attempts:{history}" if history else ""}

Based on the API results (NOT the image - you cannot see it), provide feedback:
1. Did the agent use the correct API for this question?
2. Did the agent interpret the API result correctly?
3. Are there any logical errors in the code?
4. Should the agent try a different approach?

Provide concise feedback (2-3 sentences) to help the agent improve."""

    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": prompt}],
    )

    return response.output_text


def get_vlm_reflection(
    client,
    model: str,
    image_path: str,
    question: str,
    code: str,
    api_result: dict,
    current_answer: str,
) -> str:
    """
    Condition B: VLM-guided reflection (sees the image).

    The VLM evaluator sees the image and provides grounded feedback.
    """
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    prompt = f"""You are a visual evaluator reviewing an AI agent's answer to a geometry question.

Question: {question}

The agent (who cannot see the image) wrote this code:
```python
{code}
```

API result:
{json.dumps(api_result, indent=2)}

Agent's answer: {current_answer}

YOUR TASK: Look at the image and evaluate:
1. Is the agent's answer ({current_answer}) correct based on what you see?
2. Did the API correctly detect the shapes/lines in the image?
3. If wrong, what should the correct answer be and why?

Provide concise feedback (2-3 sentences). Be specific about what you see in the image."""

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_data}",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    )

    return response.output_text


# =============================================================================
# Agent Query Logic
# =============================================================================


def query_agent(
    client,
    model: str,
    image_path: str,
    question: str,
    condition: Literal["A", "B"],
    max_rounds: int = 3,
    vlm_model: Optional[str] = None,
) -> dict:
    """
    Query the agent with self-reflection capability.

    Args:
        client: OpenAI client
        model: Model name for the agent (LLM)
        image_path: Path to image
        question: Question to answer
        condition: "A" (LLM reflection) or "B" (VLM reflection)
        max_rounds: Maximum reflection rounds
        vlm_model: Model for VLM evaluator (condition B only)

    Returns:
        dict with final_answer, reflection_rounds, function_calls, etc.
    """
    if vlm_model is None:
        vlm_model = model

    system_prompt = f"""You are a geometry analysis agent. Your task is to answer questions about geometric shapes in images.

IMPORTANT: You do NOT have direct access to the image. You must use the provided geometry_tools API functions to analyze the image.

{API_DOCUMENTATION}

Based on the question, choose the appropriate API and write code to get the answer.
"""

    # Initial prompt (no image for agent)
    user_prompt = f"""Question: {question}

Image path: {image_path}

Write Python code using the geometry_tools API to answer this question.
Your code must set:
- `answer = "YES"` or `answer = "NO"`
- `confidence = "high"`, `"medium"`, `"low"`, or `"uncertain"`

Use the execute_geometry_api tool to run your code."""

    input_list = [{"role": "user", "content": user_prompt}]

    reflection_rounds = []
    all_function_calls = []
    final_answer = ""

    for round_num in range(max_rounds):
        if benchmark_logger:
            benchmark_logger.debug(f"  === Round {round_num + 1}/{max_rounds} ===")

        # Request code from agent
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=input_list,
            tools=[CODE_EXEC_TOOL],
            tool_choice="required" if round_num == 0 else "auto",
        )

        # Add response to conversation
        input_list.extend(response.output)

        # Process function call
        for item in response.output:
            if item.type == "function_call" and item.name == "execute_geometry_api":
                arguments = json.loads(item.arguments)
                code = arguments.get("code", "")
                reasoning = arguments.get("reasoning", "")

                if benchmark_logger:
                    benchmark_logger.debug(f"  Reasoning: {reasoning}")
                    benchmark_logger.debug(f"  Code:\n{code}")

                # Execute code
                exec_result = execute_code_sandbox(code, image_path)
                all_function_calls.extend(exec_result.get("function_calls", []))

                answer = exec_result.get("answer", "")
                confidence = exec_result.get("confidence", "uncertain")

                if benchmark_logger:
                    benchmark_logger.debug(
                        f"  Result: success={exec_result['success']}, answer={answer}, confidence={confidence}"
                    )

                # Determine if we need reflection
                needs_retry = False
                reflection_feedback = ""

                if exec_result["success"]:
                    final_answer = answer

                    # Check if we should reflect (not high confidence and not last round)
                    if confidence != "high" and round_num < max_rounds - 1:
                        needs_retry = True

                        # Get reflection based on condition
                        if condition == "A":
                            reflection_feedback = get_llm_reflection(
                                client,
                                model,
                                question,
                                code,
                                exec_result,
                                [
                                    {
                                        "round": r.round_num,
                                        "code": r.code,
                                        "api_result": r.api_result,
                                        "answer": r.answer,
                                        "confidence": r.confidence,
                                    }
                                    for r in reflection_rounds
                                ],
                            )
                        else:  # Condition B
                            reflection_feedback = get_vlm_reflection(
                                client,
                                vlm_model,
                                image_path,
                                question,
                                code,
                                exec_result,
                                answer,
                            )

                        if benchmark_logger:
                            benchmark_logger.debug(
                                f"  Reflection ({condition}): {reflection_feedback}"
                            )
                else:
                    # Execution error - always retry
                    needs_retry = True if round_num < max_rounds - 1 else False
                    reflection_feedback = (
                        f"Execution error: {exec_result.get('error', 'Unknown error')}"
                    )

                # Record this round
                round_info = ReflectionRound(
                    round_num=round_num + 1,
                    code=code,
                    reasoning=reasoning,
                    api_result={
                        "success": exec_result["success"],
                        "answer": exec_result.get("answer"),
                        "confidence": exec_result.get("confidence"),
                        "output": exec_result.get("output", "")[:500],
                        "error": exec_result.get("error"),
                    },
                    answer=answer,
                    confidence=confidence,
                    reflection_feedback=reflection_feedback,
                    needs_retry=needs_retry,
                )
                reflection_rounds.append(round_info)

                # Add result to conversation
                result_msg = json.dumps(
                    {
                        "success": exec_result["success"],
                        "answer": answer,
                        "confidence": confidence,
                        "output": exec_result.get("output", ""),
                        "error": exec_result.get("error"),
                    }
                )

                input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result_msg,
                    }
                )

                # If reflection needed, add feedback to conversation
                if needs_retry and reflection_feedback:
                    if condition == "A":
                        feedback_msg = f"[Self-Reflection] Your confidence is '{confidence}'. Here's feedback to help:\n{reflection_feedback}\n\nPlease try again with a different approach or verify your answer."
                    else:
                        feedback_msg = f"[Visual Evaluator Feedback] An evaluator who can see the image says:\n{reflection_feedback}\n\nPlease reconsider your answer based on this feedback."

                    input_list.append(
                        {
                            "role": "user",
                            "content": feedback_msg,
                        }
                    )

                # If successful and high confidence (or last round), we're done
                if exec_result["success"] and (confidence == "high" or not needs_retry):
                    return {
                        "final_answer": final_answer,
                        "num_rounds": round_num + 1,
                        "reflection_rounds": reflection_rounds,
                        "function_calls": all_function_calls,
                        "condition": condition,
                    }

    # Max rounds reached
    return {
        "final_answer": final_answer,
        "num_rounds": max_rounds,
        "reflection_rounds": reflection_rounds,
        "function_calls": all_function_calls,
        "condition": condition,
    }


# =============================================================================
# Benchmark Data Loading
# =============================================================================

QUESTIONS = {
    # 1D Line comparison
    "1d_line": "Is the top horizontal line longer than the bottom horizontal line? Answer YES or NO.",
    "1d_arrow": "Is the top horizontal line (shaft) longer than the bottom horizontal line? Ignore the arrowheads. Answer YES or NO.",
    "1d_muller": "Is the top horizontal line segment longer than the bottom horizontal line segment? Ignore the arrow fins. Answer YES or NO.",
    # 2D Shape comparison
    "2d_circle": "Is the left circle larger than the right circle? Answer YES or NO.",
    "2d_square": "Is the left square larger than the right square? Answer YES or NO.",
    "2d_rectangle": "Is the left rectangle larger than the right rectangle? Answer YES or NO.",
    # 2D Triangle classification
    "2d_triangle_equilateral": "Is this a valid equilateral triangle (all three sides equal length)? Answer YES or NO.",
    "2d_triangle_right": "Is this a valid right triangle (has exactly one 90-degree angle)? Answer YES or NO.",
}


def load_benchmark_data(data_dir: str) -> list[BenchmarkSample]:
    """Load all benchmark samples from the data directory."""
    samples = []
    data_path = Path(data_dir)

    # 1D Line Length Comparison
    line_dir = data_path / "vision_1d_Compare_Length"

    for subdir, task_type in [
        ("Line_50", "1d_line"),
        ("arrow_100", "1d_arrow"),
        ("Rotated_line_50", "1d_line"),
    ]:
        folder = line_dir / subdir
        if folder.exists():
            for img in folder.glob("*.png"):
                gt = "_YES_" in img.name
                samples.append(
                    BenchmarkSample(
                        image_path=str(img),
                        task_type=task_type,
                        subtask=subdir,
                        ground_truth=gt,
                        question=QUESTIONS[task_type],
                    )
                )

    # Muller-Lyer
    muller_dir = line_dir / "muller_lyer_150"
    if muller_dir.exists():
        for img in muller_dir.glob("*.png"):
            gt = "_YES_" in img.name
            samples.append(
                BenchmarkSample(
                    image_path=str(img),
                    task_type="1d_muller",
                    subtask="muller_lyer_150",
                    ground_truth=gt,
                    question=QUESTIONS["1d_muller"],
                )
            )

    # 2D Shape Comparison
    shape_dir = data_path / "vision_2d_Compare_Size"
    for subdir, task_type in [
        ("circle_100", "2d_circle"),
        ("square_50", "2d_square"),
        ("rectangle_50", "2d_rectangle"),
        ("rotated_square_50", "2d_square"),
        ("rotated_rectangle_50", "2d_rectangle"),
    ]:
        folder = shape_dir / subdir
        if folder.exists():
            for img in folder.glob("*.png"):
                gt = "_YES_" in img.name
                samples.append(
                    BenchmarkSample(
                        image_path=str(img),
                        task_type=task_type,
                        subtask=subdir,
                        ground_truth=gt,
                        question=QUESTIONS[task_type],
                    )
                )

    # 2D Triangle Classification
    tri_dir = data_path / "vision_2d_Check_Triangles"

    eq_dir = tri_dir / "vision_equilateral_triangles_50"
    if eq_dir.exists():
        for img in eq_dir.glob("*.png"):
            gt = "_YES_" in img.name
            samples.append(
                BenchmarkSample(
                    image_path=str(img),
                    task_type="2d_triangle_equilateral",
                    subtask="equilateral_50",
                    ground_truth=gt,
                    question=QUESTIONS["2d_triangle_equilateral"],
                )
            )

    rt_dir = tri_dir / "vision_right_triangles_50"
    if rt_dir.exists():
        for img in rt_dir.glob("*.png"):
            gt = "_YES_" in img.name
            samples.append(
                BenchmarkSample(
                    image_path=str(img),
                    task_type="2d_triangle_right",
                    subtask="right_50",
                    ground_truth=gt,
                    question=QUESTIONS["2d_triangle_right"],
                )
            )

    return samples


def parse_yes_no(response: str) -> Optional[bool]:
    """Parse YES/NO from response."""
    if not response:
        return None
    response = response.strip().upper()
    if response == "YES":
        return True
    if response == "NO":
        return False
    if response.startswith("YES"):
        return True
    if response.startswith("NO"):
        return False
    return None


# =============================================================================
# Benchmark Runner
# =============================================================================


# Thread-safe counter for progress tracking
class ProgressCounter:
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.correct = 0
        self.lock = Lock()

    def increment(self, is_correct: bool):
        with self.lock:
            self.completed += 1
            if is_correct:
                self.correct += 1
            return self.completed, self.correct


def process_single_sample(
    client,
    model: str,
    sample: BenchmarkSample,
    condition: Literal["A", "B"],
    max_rounds: int,
    vlm_model: Optional[str],
    progress: ProgressCounter,
    verbose: bool,
) -> BenchmarkResult:
    """Process a single sample (no timeout - timeout handled by outer executor)."""
    start_time = time.time()
    sample_name = Path(sample.image_path).name

    try:
        query_result = query_agent(
            client=client,
            model=model,
            image_path=sample.image_path,
            question=sample.question,
            condition=condition,
            max_rounds=max_rounds,
            vlm_model=vlm_model,
        )

        predicted = parse_yes_no(query_result["final_answer"])
        is_correct = (
            (predicted == sample.ground_truth) if predicted is not None else False
        )

        result = BenchmarkResult(
            sample=sample,
            final_answer=query_result["final_answer"],
            predicted=predicted,
            correct=is_correct,
            num_rounds=query_result["num_rounds"],
            reflection_rounds=query_result["reflection_rounds"],
            functions_called=query_result["function_calls"],
            condition=condition,
        )

        elapsed = time.time() - start_time
        completed, total_correct = progress.increment(is_correct)
        status = "✓" if is_correct else "✗"
        accuracy = total_correct / completed * 100

        if verbose:
            print(
                f"[{completed}/{progress.total}] {sample_name} {status} pred={predicted} gt={sample.ground_truth} ({elapsed:.1f}s) [Acc: {accuracy:.1f}%]"
            )

        if benchmark_logger:
            benchmark_logger.info(
                f"[{completed}/{progress.total}] {sample_name} {status} pred={predicted} gt={sample.ground_truth} ({elapsed:.1f}s)"
            )
            for r in query_result.get("reflection_rounds", []):
                benchmark_logger.debug(
                    f"  Round {r.round_num}: answer={r.answer}, conf={r.confidence}"
                )

        return result

    except Exception as e:
        elapsed = time.time() - start_time
        result = BenchmarkResult(
            sample=sample,
            final_answer="",
            predicted=None,
            correct=False,
            num_rounds=0,
            error=str(e),
            condition=condition,
        )
        completed, _ = progress.increment(False)
        if verbose:
            print(
                f"[{completed}/{progress.total}] {sample_name} - ERROR: {e} ({elapsed:.1f}s)"
            )
        if benchmark_logger:
            benchmark_logger.error(
                f"[{completed}/{progress.total}] {sample_name} - ERROR: {e}"
            )
            benchmark_logger.debug(traceback.format_exc())
        return result


def run_benchmark(
    client,
    model: str,
    samples: List[BenchmarkSample],
    condition: Literal["A", "B"],
    seed: int = 42,
    max_samples: Optional[int] = None,
    max_rounds: int = 3,
    verbose: bool = True,
    vlm_model: Optional[str] = None,
    num_workers: int = 4,
    timeout: int = 120,
) -> List[BenchmarkResult]:
    """
    Run agent benchmark with specified condition.

    Args:
        client: OpenAI client
        model: Model name
        samples: List of benchmark samples
        condition: "A" or "B"
        seed: Random seed
        max_samples: Maximum samples to test
        max_rounds: Max reflection rounds per sample
        verbose: Print progress
        vlm_model: VLM model for condition B
        num_workers: Number of concurrent workers (default: 4)
        timeout: Timeout per question in seconds (default: 120)

    Returns:
        List of BenchmarkResult
    """
    random.seed(seed)
    samples = samples.copy()
    random.shuffle(samples)

    if max_samples:
        samples = samples[:max_samples]

    print(f"\nStarting benchmark with {len(samples)} samples")
    print(f"  Concurrency: {num_workers} workers")
    print(f"  Timeout: {timeout}s per question")
    print(f"  Condition: {condition}")
    print()

    progress = ProgressCounter(len(samples))

    # Process samples concurrently with timeout handling
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_sample = {
            executor.submit(
                process_single_sample,
                client=client,
                model=model,
                sample=sample,
                condition=condition,
                max_rounds=max_rounds,
                vlm_model=vlm_model,
                progress=progress,
                verbose=verbose,
            ): sample
            for sample in samples
        }

        # Collect results as they complete, with per-task timeout
        for future in as_completed(future_to_sample, timeout=None):
            sample = future_to_sample[future]
            sample_name = Path(sample.image_path).name
            try:
                # Apply per-task timeout when getting result
                result = future.result(timeout=timeout)
                results.append(result)
            except FuturesTimeoutError:
                # Task timed out
                timeout_result = BenchmarkResult(
                    sample=sample,
                    final_answer="",
                    predicted=None,
                    correct=False,
                    num_rounds=0,
                    error=f"Timeout after {timeout}s",
                    condition=condition,
                )
                results.append(timeout_result)
                completed, _ = progress.increment(False)
                if verbose:
                    print(
                        f"[{completed}/{progress.total}] {sample_name} - TIMEOUT after {timeout}s"
                    )
                if benchmark_logger:
                    benchmark_logger.warning(
                        f"[{completed}/{progress.total}] {sample_name} - TIMEOUT after {timeout}s"
                    )
            except Exception as e:
                # Other errors
                error_result = BenchmarkResult(
                    sample=sample,
                    final_answer="",
                    predicted=None,
                    correct=False,
                    num_rounds=0,
                    error=str(e),
                    condition=condition,
                )
                results.append(error_result)
                completed, _ = progress.increment(False)
                if verbose:
                    print(f"[{completed}/{progress.total}] {sample_name} - ERROR: {e}")
                if benchmark_logger:
                    benchmark_logger.error(
                        f"[{completed}/{progress.total}] {sample_name} - Future error: {e}"
                    )

    # Sort results to maintain original order (by image path)
    sample_order = {s.image_path: i for i, s in enumerate(samples)}
    results.sort(key=lambda r: sample_order.get(r.sample.image_path, 0))

    return results


# =============================================================================
# Metrics and Reporting
# =============================================================================


def compute_metrics(results: list[BenchmarkResult]) -> dict:
    """Compute accuracy metrics."""
    metrics = {
        "overall": {"correct": 0, "total": 0, "errors": 0},
        "by_task_type": defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0}),
        "by_subtask": defaultdict(lambda: {"correct": 0, "total": 0, "errors": 0}),
        "avg_rounds": 0,
        "confidence_distribution": defaultdict(int),
        "reflection_helped": 0,  # Cases where answer changed after reflection
    }

    total_rounds = 0
    for result in results:
        task_type = result.sample.task_type
        subtask = result.sample.subtask

        metrics["overall"]["total"] += 1
        metrics["by_task_type"][task_type]["total"] += 1
        metrics["by_subtask"][subtask]["total"] += 1
        total_rounds += result.num_rounds

        # Track final confidence
        if result.reflection_rounds:
            final_conf = result.reflection_rounds[-1].confidence
            metrics["confidence_distribution"][final_conf] += 1

            # Check if reflection helped (answer changed)
            if len(result.reflection_rounds) > 1:
                first_answer = result.reflection_rounds[0].answer
                final_answer = result.reflection_rounds[-1].answer
                if first_answer != final_answer:
                    metrics["reflection_helped"] += 1

        if result.error:
            metrics["overall"]["errors"] += 1
            metrics["by_task_type"][task_type]["errors"] += 1
            metrics["by_subtask"][subtask]["errors"] += 1
        elif result.correct:
            metrics["overall"]["correct"] += 1
            metrics["by_task_type"][task_type]["correct"] += 1
            metrics["by_subtask"][subtask]["correct"] += 1

    def add_accuracy(d):
        if d["total"] > 0:
            d["accuracy"] = d["correct"] / d["total"]
        else:
            d["accuracy"] = 0.0

    add_accuracy(metrics["overall"])
    for m in metrics["by_task_type"].values():
        add_accuracy(m)
    for m in metrics["by_subtask"].values():
        add_accuracy(m)

    if results:
        metrics["avg_rounds"] = total_rounds / len(results)

    return metrics


def print_report(metrics: dict, model_name: str, condition: str):
    """Print benchmark report."""
    print("\n" + "=" * 70)
    print("AGENT BENCHMARK V2 REPORT")
    print(f"Model: {model_name} | Condition: {condition}")
    print("=" * 70)

    overall = metrics["overall"]
    print(
        f"\nOverall Accuracy: {overall['accuracy']:.2%} ({overall['correct']}/{overall['total']})"
    )
    print(f"Average Rounds: {metrics['avg_rounds']:.2f}")
    print(f"Reflection Helped (answer changed): {metrics['reflection_helped']} cases")

    if overall["errors"] > 0:
        print(f"Errors: {overall['errors']}")

    print("\n" + "-" * 50)
    print("Confidence Distribution:")
    print("-" * 50)
    for conf, count in sorted(metrics["confidence_distribution"].items()):
        print(f"  {conf:12s}: {count:3d}")

    print("\n" + "-" * 50)
    print("Accuracy by Task Type:")
    print("-" * 50)

    task_order = [
        "1d_line",
        "1d_arrow",
        "1d_muller",
        "2d_circle",
        "2d_square",
        "2d_rectangle",
        "2d_triangle_equilateral",
        "2d_triangle_right",
    ]

    for task_type in task_order:
        if task_type in metrics["by_task_type"]:
            m = metrics["by_task_type"][task_type]
            print(
                f"  {task_type:30s}: {m['accuracy']:6.2%} ({m['correct']:3d}/{m['total']:3d})"
            )

    print("=" * 70)


def save_results(
    results: list[BenchmarkResult],
    metrics: dict,
    output_path: str,
    condition: str,
    model: str,
    seed: int,
):
    """Save results to JSON."""
    output = {
        "metadata": {
            "model": model,
            "condition": condition,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
        },
        "metrics": {
            "overall": metrics["overall"],
            "by_task_type": dict(metrics["by_task_type"]),
            "by_subtask": dict(metrics["by_subtask"]),
            "avg_rounds": metrics["avg_rounds"],
            "confidence_distribution": dict(metrics["confidence_distribution"]),
            "reflection_helped": metrics["reflection_helped"],
        },
        "results": [
            {
                "image": result.sample.image_path,
                "task_type": result.sample.task_type,
                "subtask": result.sample.subtask,
                "question": result.sample.question,
                "ground_truth": result.sample.ground_truth,
                "final_answer": result.final_answer,
                "predicted": result.predicted,
                "correct": result.correct,
                "num_rounds": result.num_rounds,
                "condition": result.condition,
                "error": result.error,
                "reflection_rounds": [
                    {
                        "round_num": r.round_num,
                        "code": r.code,
                        "reasoning": r.reasoning,
                        "api_result": r.api_result,
                        "answer": r.answer,
                        "confidence": r.confidence,
                        "reflection_feedback": r.reflection_feedback,
                        "needs_retry": r.needs_retry,
                    }
                    for r in result.reflection_rounds
                ],
                "functions_called": result.functions_called,
            }
            for result in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# =============================================================================
# Constants
# =============================================================================

# Fixed model for all experiments (same as v1)
MODEL_NAME = "gpt-5"


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Agent Benchmark V2 with Self-Reflection (Model: GPT-5)"
    )
    parser.add_argument(
        "--condition",
        "-c",
        type=str,
        choices=["A", "B"],
        required=True,
        help="Experiment condition: A (LLM-only reflection, no image) or B (VLM-guided reflection, evaluator sees image)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-samples",
        "-n",
        type=int,
        default=None,
        help="Max samples to test",
    )
    parser.add_argument(
        "--max-rounds",
        "-r",
        type=int,
        default=3,
        help="Max reflection rounds per sample (default: 3)",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default="2821_finalProj_IMAGE_DATA",
        help="Data directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "--log",
        "-l",
        type=str,
        default=None,
        help="Log file path",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=4,
        help="Number of concurrent workers (default: 4)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=120,
        help="Timeout per question in seconds (default: 120)",
    )

    args = parser.parse_args()

    # Fixed model
    model = MODEL_NAME

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = args.log or f"benchmark_v2_{args.condition}_{timestamp}.log"
    setup_logging(log_file)

    benchmark_logger.info(f"Benchmark V2 started at {datetime.now().isoformat()}")
    benchmark_logger.info(f"Model: {model} (fixed)")
    benchmark_logger.info(f"Condition: {args.condition}")
    benchmark_logger.info(f"Seed: {args.seed}")
    benchmark_logger.info(f"Max rounds: {args.max_rounds}")
    benchmark_logger.info(f"Workers: {args.workers}")
    benchmark_logger.info(f"Timeout: {args.timeout}s")

    # Initialize client
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Load data
    print(f"Loading benchmark data from: {args.data_dir}")
    samples = load_benchmark_data(args.data_dir)
    print(f"Loaded {len(samples)} samples")

    condition_desc = {
        "A": "LLM-only reflection (GPT-5 without image)",
        "B": "VLM-guided reflection (GPT-5 with image as evaluator)",
    }

    print("\nExperiment Configuration:")
    print(f"  Model: {model} (fixed)")
    print(f"  Condition: {args.condition} - {condition_desc[args.condition]}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Workers: {args.workers}")
    print(f"  Timeout: {args.timeout}s per question")
    print(f"  Logging to: {log_file}")

    # Run benchmark
    results = run_benchmark(
        client=client,
        model=model,
        samples=samples,
        condition=args.condition,
        seed=args.seed,
        max_samples=args.max_samples,
        max_rounds=args.max_rounds,
        verbose=not args.quiet,
        vlm_model=model,  # Same model for VLM evaluator
        num_workers=args.workers,
        timeout=args.timeout,
    )

    # Report
    metrics = compute_metrics(results)
    print_report(metrics, model, args.condition)

    # Save
    output_file = args.output or f"benchmark_v2_{args.condition}_{args.seed}.json"
    save_results(results, metrics, output_file, args.condition, model, args.seed)

    print(f"Log saved to: {log_file}")


if __name__ == "__main__":
    main()
