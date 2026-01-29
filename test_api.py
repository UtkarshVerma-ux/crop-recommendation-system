"""
Check installed package versions
"""

import sys

packages = {
    'API': [
        'fastapi',
        'uvicorn',
        'pydantic',
        'pandas',
        'numpy',
        'scikit-learn',
        'sklearn',
        'xgboost',
        'catboost',
        'requests',
        'python-multipart'
    ],
    'Interface': [
        'streamlit',
        'requests',
        'pandas',
        'plotly',
        'numpy'
    ]
}

def get_version(package_name):
    """Get version of installed package"""
    try:
        if package_name == 'sklearn':
            import sklearn
            return sklearn.__version__
        else:
            module = __import__(package_name)
            return module.__version__
    except ImportError:
        return "NOT INSTALLED"
    except AttributeError:
        try:
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except:
            return "VERSION UNKNOWN"

print("="*70)
print("📦 INSTALLED PACKAGE VERSIONS")
print("="*70)

for category, pkg_list in packages.items():
    print(f"\n{'='*70}")
    print(f"📂 {category} Requirements:")
    print(f"{'='*70}")
    
    for pkg in pkg_list:
        version = get_version(pkg)
        status = "✅" if version not in ["NOT INSTALLED", "VERSION UNKNOWN"] else "❌"
        print(f"{status} {pkg:20s} {version}")

print("\n" + "="*70)
print("📝 Copy the versions above to your requirements.txt files!")
print("="*70)