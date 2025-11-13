import functools
import warnings
import inspect
from typing import Callable, Set, Any
from packages.train.training import Trainer

def requires_trainer_attrs(*required_attrs: str):
    """
    Decorator that checks if a Trainer object has required attributes before executing a function.
    
    Args:
        *required_attrs: Attribute names that must exist on the trainer object
    
    Usage:
        @requires_trainer_attrs('model', 'train_loader', 'val_loader')
        def plot_training_curves(trainer):
            # Function implementation
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Find the trainer argument
            trainer = None
            
            # Check positional args
            sig = inspect.signature(func)
            param_names = list(sig.parameters.keys())
            
            # Look for 'trainer' in args by position or name
            for i, arg in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                    if param_name == 'trainer' or isinstance(arg, Trainer):
                        trainer = arg
                        break
            
            # Check kwargs
            if trainer is None and 'trainer' in kwargs:
                trainer = kwargs['trainer']
            
            # If no trainer found, try to find any Trainer instance
            if trainer is None:
                for arg in args:
                    if isinstance(arg, Trainer):
                        trainer = arg
                        break
            
            if trainer is None:
                warnings.warn(
                    f"Could not find Trainer object in arguments for {func.__name__}. "
                    f"Function not executed.",
                    UserWarning,
                    stacklevel=2
                )
                return None
            
            # Check for required attributes
            missing_attrs = []
            for attr in required_attrs:
                if not hasattr(trainer, attr):
                    missing_attrs.append(attr)
            
            if missing_attrs:
                warnings.warn(
                    f"Trainer object missing required attributes for {func.__name__}: "
                    f"{', '.join(missing_attrs)}. Function not executed.",
                    UserWarning,
                    stacklevel=2
                )
                return None
            
            # All checks passed, execute the function
            try:
                return func(*args, **kwargs)
            except Exception as e:
                warnings.warn(
                    f"Error executing {func.__name__}: {str(e)}. "
                    f"Continuing execution...",
                    UserWarning,
                    stacklevel=2
                )
                return None
        
        return wrapper
    return decorator


# Alternative: Auto-detect required attributes from function body
def auto_validate_trainer_attrs(func: Callable) -> Callable:
    """
    Decorator that automatically detects which trainer attributes are used
    in the function and validates they exist.
    
    Usage:
        @auto_validate_trainer_attrs
        def plot_training_curves(trainer):
            # Uses trainer.model, trainer.train_loader automatically detected
            pass
    """
    import ast
    import textwrap
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Find trainer argument (same logic as above)
        trainer = None
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
        
        for i, arg in enumerate(args):
            if i < len(param_names):
                param_name = param_names[i]
                if param_name == 'trainer' or isinstance(arg, Trainer):
                    trainer = arg
                    break
        
        if trainer is None and 'trainer' in kwargs:
            trainer = kwargs['trainer']
        
        if trainer is None:
            for arg in args:
                if isinstance(arg, Trainer):
                    trainer = arg
                    break
        
        if trainer is None:
            warnings.warn(
                f"Could not find Trainer object for {func.__name__}",
                UserWarning,
                stacklevel=2
            )
            return None
        
        # Parse function source to find trainer attribute accesses
        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
            tree = ast.parse(source)
            
            required_attrs = set()
            
            class TrainerAttributeVisitor(ast.NodeVisitor):
                def visit_Attribute(self, node):
                    # Look for patterns like trainer.model, trainer.train_loader
                    if isinstance(node.value, ast.Name):
                        # Check if it's the trainer parameter
                        if node.value.id in param_names:
                            # Assume first param is trainer if named 'trainer'
                            if node.value.id == 'trainer' or param_names.index(node.value.id) == 0:
                                required_attrs.add(node.attr)
                    self.generic_visit(node)
            
            visitor = TrainerAttributeVisitor()
            visitor.visit(tree)
            
            # Check for missing attributes
            missing_attrs = [attr for attr in required_attrs if not hasattr(trainer, attr)]
            
            if missing_attrs:
                warnings.warn(
                    f"Trainer missing attributes for {func.__name__}: "
                    f"{', '.join(missing_attrs)}",
                    UserWarning,
                    stacklevel=2
                )
                return None
            
        except Exception as e:
            # If parsing fails, just try to execute
            warnings.warn(
                f"Could not validate attributes for {func.__name__}: {e}",
                UserWarning,
                stacklevel=2
            )
        
        # Execute function with error handling
        try:
            return func(*args, **kwargs)
        except AttributeError as e:
            warnings.warn(
                f"AttributeError in {func.__name__}: {e}. "
                f"Check trainer attributes.",
                UserWarning,
                stacklevel=2
            )
            return None
        except Exception as e:
            warnings.warn(
                f"Error in {func.__name__}: {e}",
                UserWarning,
                stacklevel=2
            )
            return None
    
    return wrapper


# Simpler version: Just catch errors and warn
def safe_plot(func: Callable) -> Callable:
    """
    Simple decorator that catches all errors in plotting functions and warns instead of crashing.
    
    Usage:
        @safe_plot
        def plot_something(trainer):
            trainer.model.plot()
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AttributeError as e:
            warnings.warn(
                f"AttributeError in {func.__name__}: {e}. "
                f"Required trainer attributes may be missing.",
                UserWarning,
                stacklevel=2
            )
            return None
        except Exception as e:
            warnings.warn(
                f"Error executing {func.__name__}: {type(e).__name__}: {e}",
                UserWarning,
                stacklevel=2
            )
            return None
    
    return wrapper