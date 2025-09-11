import os
import sys

def find_trame_vtk_js():
    # 현재 실행 중인 가상 환경의 site-packages 경로를 찾습니다.
    # LLMvenv\Lib\site-packages
    site_packages = os.path.normpath(os.path.join(sys.prefix, 'Lib', 'site-packages'))
    
    if not os.path.exists(site_packages):
        print(f"Error: site-packages directory not found at {site_packages}")
        return

    print(f"Searching for 'trame-vtk.js' in: {site_packages}")
    
    found_path = None
    for root, dirs, files in os.walk(site_packages):
        if 'trame-vtk.js' in files:
            found_path = os.path.join(root)
            break
            
    if found_path:
        print(f"\n✅ Found 'trame-vtk.js' at:")
        print(found_path)
    else:
        print("\n❌ Could not find 'trame-vtk.js'. Please check your 'trame-vtk' installation.")

if __name__ == "__main__":
    find_trame_vtk_js()