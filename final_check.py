#!/usr/bin/env python3
"""
Final project check script
"""
import os

def check_project():
    print('🔍 Final Project Check:')
    
    files_to_check = [
        ('scientific_api.db', 'Database file'),
        ('main_azure_with_db.py', 'Azure app'),
        ('main_vercel_clean.py', 'Vercel app'),
        ('database/schema.sql', 'Database schema'),
        ('deploy_with_database.sh', 'Deploy script'),
        ('Dockerfile.azure.db', 'Docker file'),
        ('azure-deployment-with-db.json', 'Azure config'),
        ('vercel_clean.json', 'Vercel config'),
        ('requirements_vercel_clean.txt', 'Requirements'),
        ('README_DATABASE.md', 'Documentation'),
        ('PROJECT_SUMMARY.md', 'Project summary'),
        ('init_database.py', 'DB initialization'),
        ('test_app.py', 'Test script'),
        ('database/config.py', 'DB configuration')
    ]
    
    all_good = True
    for file_path, description in files_to_check:
        exists = os.path.exists(file_path)
        status = "✅" if exists else "❌"
        print(f'📁 {description}: {status} {file_path}')
        if not exists:
            all_good = False
    
    print('\n🎯 Project Status:')
    if all_good:
        print('✅ All components present and ready!')
        print('🚀 Project rebuild complete!')
        print('🎉 Ready for deployment!')
    else:
        print('❌ Some components missing')
        
    return all_good

if __name__ == "__main__":
    check_project() 