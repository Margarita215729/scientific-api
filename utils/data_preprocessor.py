"""
Data preprocessor for astronomical catalogs.
Automatically downloads, cleans and normalizes astronomical datasets on container startup.
"""

import os
import pandas as pd
import numpy as np
import logging
import asyncio
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AstronomicalDataPreprocessor:
    """Preprocessor for astronomical catalogs from SDSS, DESI, DES, and Euclid."""
    
    def __init__(self, data_dir: str = "galaxy_data"):
        self.data_dir = Path(data_dir)
        self.cache_dir = self.data_dir / "cache"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Catalog configurations with alternative column names
        self.catalogs = {
            "SDSS": {
                "url": "https://dr18.sdss.org/sas/dr18/spectro/boss/redux/v6_0_4/spAll-v6_0_4.fits",
                "columns": {
                    "ra": ["PLUG_RA", "RA", "RAJ2000", "ALPHA_J2000"],
                    "dec": ["PLUG_DEC", "DEC", "DECJ2000", "DEJ2000", "DELTA_J2000"],
                    "z": ["Z", "REDSHIFT", "PHOTOZ", "Z_MEAN", "Z_SPEC", "ZPHOT"],
                    "z_err": ["Z_ERR", "ZERR", "Z_ERR_SPEC", "REDSHIFT_ERR"],
                    "mag_g": ["MAG", "PSFMAG_G", "FIBERMAG_G", "MAG_G"],
                    "mag_r": ["MAG", "PSFMAG_R", "FIBERMAG_R", "MAG_R"],
                    "mag_i": ["MAG", "PSFMAG_I", "FIBERMAG_I", "MAG_I"]
                },
                "processed_name": "sdss_processed.csv",
                "sample_size": 50000
            },
            "DESI": {
                "url": "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/zcatalog/zall-pix-fuji.fits",
                "columns": {
                    "ra": ["TARGET_RA", "RA", "RAJ2000"],
                    "dec": ["TARGET_DEC", "DEC", "DECJ2000"],
                    "z": ["Z", "REDSHIFT", "Z_SPEC"],
                    "z_err": ["ZERR", "Z_ERR", "REDSHIFT_ERR"],
                    "mag_g": ["FLUX_G", "MAG_G", "PSFMAG_G"],
                    "mag_r": ["FLUX_R", "MAG_R", "PSFMAG_R"],
                    "mag_z": ["FLUX_Z", "MAG_Z", "PSFMAG_Z"]
                },
                "processed_name": "desi_processed.csv",
                "sample_size": 30000
            },
            "DES": {
                "url": "https://desdr-server.ncsa.illinois.edu/despublic/y6a2_files/y6_gold/Y6_GOLD_2_2-519.json",
                "columns": {
                    "ra": ["ALPHAWIN_J2000", "RA", "RAJ2000", "ALPHA_J2000"],
                    "dec": ["DELTAWIN_J2000", "DEC", "DECJ2000", "DELTA_J2000"],
                    "z": ["DNF_ZMEAN_SOF", "Z_PHOT", "PHOTOZ", "Z_MEAN", "REDSHIFT"],
                    "z_err": ["DNF_ZSIGMA_SOF", "Z_PHOT_ERR", "PHOTOZ_ERR", "Z_ERR"],
                    "mag_g": ["WAVG_MAG_PSF_G", "MAG_AUTO_G", "MAG_G"],
                    "mag_r": ["WAVG_MAG_PSF_R", "MAG_AUTO_R", "MAG_R"],
                    "mag_i": ["WAVG_MAG_PSF_I", "MAG_AUTO_I", "MAG_I"]
                },
                "processed_name": "des_processed.csv",
                "sample_size": 40000
            },
            "Euclid": {
                "url": "https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/catalogs/MER_FINAL_CATALOG/102018211/EUC_MER_FINAL-CAT_TILE102018211-CC66F6_20241018T214045.289017Z_00.00.fits",
                "columns": {
                    "ra": ["RIGHT_ASCENSION", "RA", "RAJ2000", "ALPHA_J2000"],
                    "dec": ["DECLINATION", "DEC", "DECJ2000", "DELTA_J2000"],
                    "z": ["Z_PHOT", "PHOTOZ", "Z_B", "REDSHIFT"],
                    "z_err": ["Z_PHOT_ERR", "PHOTOZ_ERR", "Z_ERR"],
                    "object_id": ["OBJECT_ID", "ID", "SOURCE_ID"],
                    "mag_vis": ["FLUX_VIS_1FWHM_APER", "MAG_VIS", "VIS_MAG"],
                    "mag_y": ["FLUX_Y_1FWHM_APER", "MAG_Y", "Y_MAG"],
                    "mag_j": ["FLUX_J_1FWHM_APER", "MAG_J", "J_MAG"],
                    "mag_h": ["FLUX_H_1FWHM_APER", "MAG_H", "H_MAG"]
                },
                "processed_name": "euclid_processed.csv",
                "sample_size": 30000
            }
        }
    
    def find_column_name(self, available_columns: List[str], possible_names: List[str]) -> Optional[str]:
        """Find the first matching column name from a list of possibilities."""
        available_upper = [col.upper() for col in available_columns]
        for name in possible_names:
            if name.upper() in available_upper:
                # Return the original case column name
                idx = available_upper.index(name.upper())
                return available_columns[idx]
        return None
    
    async def download_catalog(self, catalog_name: str, catalog_info: Dict) -> Optional[str]:
        """Download a catalog file."""
        url = catalog_info["url"]
        if not url:
            logger.warning(f"No URL provided for {catalog_name}")
            return None
            
        # Determine file extension from URL
        if url.endswith('.fits'):
            filename = f"{catalog_name.lower()}_raw.fits"
        elif url.endswith('.h5'):
            filename = f"{catalog_name.lower()}_raw.h5"
        elif url.endswith('.parquet'):
            filename = f"{catalog_name.lower()}_raw.parquet"
        else:
            filename = f"{catalog_name.lower()}_raw.fits"  # Default to FITS
            
        file_path = self.cache_dir / filename
        
        if file_path.exists():
            logger.info(f"Using cached {catalog_name} data: {file_path}")
            return str(file_path)
        
        logger.info(f"Downloading {catalog_name} from {url}")
        
        try:
            if AIOHTTP_AVAILABLE:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            with open(file_path, 'wb') as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    f.write(chunk)
                            logger.info(f"Downloaded {catalog_name} to {file_path}")
                            return str(file_path)
                        else:
                            logger.error(f"Failed to download {catalog_name}: HTTP {response.status}")
                            return None
            else:
                # Fallback to requests for synchronous download
                import requests
                response = requests.get(url, stream=True, timeout=300)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Downloaded {catalog_name} to {file_path}")
                    return str(file_path)
                else:
                    logger.error(f"Failed to download {catalog_name}: HTTP {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"Error downloading {catalog_name}: {e}")
            return None

    def process_catalog(self, catalog_name: str, catalog_info: Dict, file_path: str) -> Optional[str]:
        """Process and normalize a catalog."""
        try:
            logger.info(f"Processing {catalog_name}...")
            
            # Try to read FITS file
            try:
                if file_path.endswith('.parquet'):
                    # Handle Parquet files (like DES)
                    df = pd.read_parquet(file_path)
                    logger.info(f"Read Parquet file with {len(df)} rows and {len(df.columns)} columns")
                elif file_path.endswith('.h5'):
                    # Handle H5 files
                    import h5py
                    with h5py.File(file_path, 'r') as f:
                        # Find the main dataset
                        datasets = []
                        f.visititems(lambda name, obj: datasets.append(name) if isinstance(obj, h5py.Dataset) else None)
                        
                        if datasets:
                            main_dataset = datasets[0]  # Use first dataset
                            logger.info(f"Reading H5 dataset: {main_dataset}")
                            data = f[main_dataset][:]
                            df = pd.DataFrame(data)
                        else:
                            logger.error("No datasets found in H5 file")
                            return None
                else:
                    # Handle FITS files
                    try:
                        from astropy.io import fits
                        from astropy.table import Table
                    except ImportError:
                        logger.error("astropy not available - cannot process FITS files")
                        return None
                    with fits.open(file_path) as hdul:
                        # Try different extensions
                        for i, hdu in enumerate(hdul):
                            if hasattr(hdu, 'data') and hdu.data is not None:
                                logger.info(f"Reading data from HDU {i}")
                                # Convert astropy table to pandas DataFrame
                                table = Table(hdu.data)
                                
                                # Filter out multidimensional columns
                                simple_columns = []
                                for col_name in table.colnames:
                                    col = table[col_name]
                                    if len(col.shape) == 1:  # Only 1D columns
                                        simple_columns.append(col_name)
                                    else:
                                        logger.debug(f"Skipping multidimensional column: {col_name} with shape {col.shape}")
                                
                                # Create DataFrame with only simple columns
                                simple_table = table[simple_columns]
                                df = simple_table.to_pandas()
                                break
                        else:
                            logger.error("No valid data found in FITS file")
                            return None
            except ImportError:
                logger.error("astropy not available - cannot process FITS files")
                return None
            except Exception as e:
                logger.error(f"Error reading file: {e}")
                return None
            
            # Normalize column names using the mapping
            df = self.normalize_columns(df, catalog_name, catalog_info)
            if df is None:
                return None
            
            # Clean and filter data
            df = self.clean_data(df)
            
            # Sample data if too large
            sample_size = catalog_info.get("sample_size", 50000)
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
                logger.info(f"Sampled {sample_size} objects from {catalog_name}")
            
            # Add catalog source
            df['catalog'] = catalog_name
            
            # Save processed data
            output_path = self.processed_dir / catalog_info["processed_name"]
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed {catalog_name} data: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error processing {catalog_name}: {e}")
            return None
    
    def normalize_columns(self, df: pd.DataFrame, catalog_name: str, catalog_info: Dict) -> Optional[pd.DataFrame]:
        """Normalize column names using the mapping."""
        try:
            available_columns = list(df.columns)
            logger.info(f"Available columns in {catalog_name}: {available_columns[:10]}...")  # Show first 10
            
            column_mapping = {}
            required_columns = ['ra', 'dec', 'z']
            
            for standard_name, possible_names in catalog_info["columns"].items():
                found_column = self.find_column_name(available_columns, possible_names)
                if found_column:
                    column_mapping[found_column] = standard_name
                    logger.info(f"Mapped {found_column} -> {standard_name}")
                elif standard_name in required_columns:
                    logger.error(f"Required column '{standard_name}' not found in {catalog_name}")
                    logger.error(f"Looked for: {possible_names}")
                    return None
            
            if not column_mapping:
                logger.error(f"No columns could be mapped for {catalog_name}")
                return None
            
            # Rename columns
            df_normalized = df.rename(columns=column_mapping)
            
            # Keep only mapped columns
            mapped_columns = list(column_mapping.values())
            df_normalized = df_normalized[mapped_columns]
            
            logger.info(f"Normalized {catalog_name} columns: {mapped_columns}")
            return df_normalized
            
        except Exception as e:
            logger.error(f"Error normalizing columns for {catalog_name}: {e}")
            return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate astronomical data."""
        initial_count = len(df)
        
        # Remove rows with invalid coordinates
        df = df.dropna(subset=['ra', 'dec'])
        df = df[(df['ra'] >= 0) & (df['ra'] <= 360)]
        df = df[(df['dec'] >= -90) & (df['dec'] <= 90)]
        
        # Remove rows with invalid redshifts
        if 'z' in df.columns:
            df = df.dropna(subset=['z'])
            df = df[(df['z'] >= 0) & (df['z'] <= 10)]  # Reasonable redshift range
        
        # Remove rows with invalid magnitudes (if present)
        mag_columns = [col for col in df.columns if col.startswith('mag_')]
        for mag_col in mag_columns:
            if mag_col in df.columns:
                df = df.dropna(subset=[mag_col])
                df = df[(df[mag_col] > 0) & (df[mag_col] < 30)]  # Reasonable magnitude range
        
        cleaned_count = len(df)
        logger.info(f"Cleaned data: {initial_count} -> {cleaned_count} objects")
        
        return df
    
    def add_derived_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived astronomical properties."""
        try:
            # Add Cartesian coordinates (simplified, assuming flat cosmology)
            if 'z' in df.columns:
                # Simple distance calculation (not cosmologically accurate)
                c = 299792.458  # km/s
                H0 = 70  # km/s/Mpc
                df['distance_mpc'] = (c * df['z']) / H0
                
                # Convert to Cartesian coordinates
                ra_rad = np.radians(df['ra'])
                dec_rad = np.radians(df['dec'])
                
                df['x'] = df['distance_mpc'] * np.cos(dec_rad) * np.cos(ra_rad)
                df['y'] = df['distance_mpc'] * np.cos(dec_rad) * np.sin(ra_rad)
                df['z_coord'] = df['distance_mpc'] * np.sin(dec_rad)
            
            # Add color indices if multiple magnitudes available
            mag_columns = [col for col in df.columns if col.startswith('mag_')]
            if len(mag_columns) >= 2:
                for i in range(len(mag_columns) - 1):
                    mag1, mag2 = mag_columns[i], mag_columns[i + 1]
                    color_name = f"color_{mag1.split('_')[1]}_{mag2.split('_')[1]}"
                    df[color_name] = df[mag1] - df[mag2]
            
            logger.info("Added derived properties")
            return df
            
        except Exception as e:
            logger.error(f"Error adding derived properties: {e}")
            return df
    
    async def preprocess_all_catalogs(self) -> Dict[str, Any]:
        """Preprocess all available catalogs."""
        logger.info("Starting preprocessing of all astronomical catalogs...")
        
        results = {
            "status": "completed",
            "catalogs": {},
            "processed_at": datetime.now().isoformat(),
            "total_objects": 0
        }
        
        for catalog_name, catalog_info in self.catalogs.items():
            try:
                logger.info(f"Processing {catalog_name}...")
                
                # Download catalog
                if catalog_info["url"]:
                    file_path = await self.download_catalog(catalog_name, catalog_info)
                    if not file_path:
                        logger.error(f"Failed to download {catalog_name}")
                        results["catalogs"][catalog_name] = {"status": "download_failed"}
                        continue
                else:
                    logger.error(f"No URL provided for {catalog_name}")
                    results["catalogs"][catalog_name] = {"status": "no_url"}
                    continue
                
                # Process catalog
                processed_path = self.process_catalog(catalog_name, catalog_info, file_path)
                if not processed_path:
                    logger.error(f"Failed to process {catalog_name}")
                    results["catalogs"][catalog_name] = {"status": "processing_failed"}
                    continue
                
                # Load processed data to get statistics
                df = pd.read_csv(processed_path)
                df = self.add_derived_properties(df)
                df.to_csv(processed_path, index=False)  # Save with derived properties
                
                results["catalogs"][catalog_name] = {
                    "status": "success",
                    "objects": len(df),
                    "file": processed_path,
                    "columns": list(df.columns)
                }
                results["total_objects"] += len(df)
                
                logger.info(f"Successfully processed {catalog_name}: {len(df)} objects")
                
            except Exception as e:
                logger.error(f"Error processing {catalog_name}: {e}")
                results["catalogs"][catalog_name] = {"status": "error", "error": str(e)}
        
        # Create merged dataset
        if results["total_objects"] > 0:
            merged_path = await self.create_merged_dataset()
            if merged_path:
                results["merged_dataset"] = merged_path
        
        logger.info(f"Preprocessing completed. Total objects: {results['total_objects']}")
        return results
    
    async def create_merged_dataset(self) -> Optional[str]:
        """Create a merged dataset from all processed catalogs."""
        try:
            logger.info("Creating merged dataset...")
            
            all_dataframes = []
            for catalog_name, catalog_info in self.catalogs.items():
                processed_path = self.processed_dir / catalog_info["processed_name"]
                if processed_path.exists():
                    df = pd.read_csv(processed_path)
                    all_dataframes.append(df)
                    logger.info(f"Added {len(df)} objects from {catalog_name}")
            
            if not all_dataframes:
                logger.warning("No processed catalogs found for merging")
                return None
            
            # Merge all dataframes
            merged_df = pd.concat(all_dataframes, ignore_index=True)
            
            # Remove duplicates based on object_id and coordinates
            initial_count = len(merged_df)
            
            # First, remove exact duplicates by object_id if available
            if 'object_id' in merged_df.columns:
                merged_df = merged_df.drop_duplicates(subset=['object_id'], keep='first')
                logger.info(f"Removed {initial_count - len(merged_df)} exact duplicates by object_id")
            
            # Then remove coordinate-based duplicates (within 1 arcsec)
            current_count = len(merged_df)
            merged_df['ra_rounded'] = merged_df['ra'].round(4)  # ~0.36 arcsec precision
            merged_df['dec_rounded'] = merged_df['dec'].round(4)
            merged_df = merged_df.drop_duplicates(subset=['ra_rounded', 'dec_rounded'], keep='first')
            merged_df = merged_df.drop(['ra_rounded', 'dec_rounded'], axis=1)
            
            final_count = len(merged_df)
            coordinate_removed = current_count - final_count
            total_removed = initial_count - final_count
            
            if coordinate_removed > 0:
                logger.info(f"Removed {coordinate_removed} coordinate-based duplicates")
            logger.info(f"Total duplicates removed: {total_removed} ({total_removed/initial_count*100:.1f}%)")
            
            # Save merged dataset
            merged_path = self.processed_dir / "merged_catalog.csv"
            merged_df.to_csv(merged_path, index=False)
            logger.info(f"Saved merged dataset: {merged_path} ({final_count} objects)")
            
            return str(merged_path)
            
        except Exception as e:
            logger.error(f"Error creating merged dataset: {e}")
            return None

async def main():
    """Main preprocessing function."""
    preprocessor = AstronomicalDataPreprocessor()
    results = await preprocessor.preprocess_all_catalogs()
    
    print("\n" + "="*50)
    print("PREPROCESSING RESULTS")
    print("="*50)
    print(f"Status: {results['status']}")
    print(f"Total objects: {results['total_objects']:,}")
    print(f"Processed at: {results['processed_at']}")
    
    print("\nCatalog Results:")
    for catalog, info in results['catalogs'].items():
        status = info['status']
        if status == 'success':
            print(f"  {catalog}: ✅ {info['objects']:,} objects")
        else:
            print(f"  {catalog}: ❌ {status}")
    
    if 'merged_dataset' in results:
        print(f"\nMerged dataset: {results['merged_dataset']}")

if __name__ == "__main__":
    asyncio.run(main()) 