from pathlib import Path, PureWindowsPath
import functools
from datetime import datetime
import os

def log_function_call(log_file):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get the current date and time
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            
            Wpath = PureWindowsPath(args[0])
            path = Path(Wpath)
            # Combine positional and keyword arguments
            all_args = ', '.join(
                [f"{repr(arg)}" for arg in args] +
                [f"{key}={repr(value)}" for key, value in kwargs.items()]
            )

            # Create the log entry
            log_entry = f"{current_time}: {func.__name__}({all_args})\n"

            # Write the log entry to the specified log file

            filename = str(log_file) + str(current_time) + ".txt"

            log_path = path / filename.replace(":", "_")
            with open(log_path, 'a') as file:
                file.write(log_entry)

            # Call the wrapped function and return its result
            result = func(*args, **kwargs)
            return result

        return wrapper
    return decorator