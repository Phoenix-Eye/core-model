import numpy as np
from pathlib import Path

def verify_data_structure():
    """Verify that all required data files exist and have correct format"""
    try:
        # Check processed data
        data_path = Path('data/processed/processed_data.npz')
        if not data_path.exists():
            return False, "Processed data not found"
            
        data = np.load(data_path)
        required_keys = ['spatial', 'temporal', 'labels']
        if not all(key in data.files for key in required_keys):
            return False, "Processed data missing required arrays"
            
        # Check shapes
        if data['spatial'].shape[1:] != (64, 64, 5):
            return False, "Incorrect spatial data shape"
        if data['temporal'].shape[1:] != (24, 10):
            return False, "Incorrect temporal data shape"
        if data['labels'].shape[1:] != (64, 64):
            return False, "Incorrect labels shape"
            
        return True, "Data verification successful"
    except Exception as e:
        return False, f"Verification failed: {str(e)}"

if __name__ == '__main__':
    success, message = verify_data_structure()
    print(message)
    exit(0 if success else 1)