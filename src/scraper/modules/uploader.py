import logging
import os


def save_new_files(file_paths, storage_dir):
    """
    Moves new files to the storage directory.
    WILL REQUIRE MODIFICATION FOR ACTUAL UPLOADING TO A REMOTE SERVER.
    """

    os.makedirs(storage_dir, exist_ok=True)
    saved_files = []
    for path in file_paths:
        filename = os.path.basename(path)
        dest_path = os.path.join(storage_dir, filename)
        if not os.path.exists(dest_path):
            os.rename(path, dest_path)
            saved_files.append(dest_path)
            logging.info("Uploaded new file: %s", dest_path)
    return saved_files
