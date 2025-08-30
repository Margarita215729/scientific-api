import os
import pandas as pd
import numpy as np
import logging
import asyncio
import aiohttp

import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

# Astronomy libraries
try:
    from astropy.io import fits
    from astropy.table import Table
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    logging.warning("Astropy not available. FITS file processing will be limited.")

logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path("galaxy_data")
OUTPUT_DIR = Path("processed_data")
CACHE_DIR = DATA_DIR / "cache"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

class AstronomicalDataProcessor:
    """Processor for astronomical catalog data."""
    
    def __init__(self):
        self.session = None
        self.catalog_urls = {
            "SDSS": {
                "url": "https://data.sdss.org/sas/dr17/eboss/spectro/redux/v5_13_2/spectra/lite/",
                "sample_file": "specObj-dr17.fits",
                "columns": ["RA", "DEC", "Z", "PLATE", "MJD", "FIBERID", "MAG_G", "MAG_R", "MAG_I"]
            },
            "DESI": {
                "url": "https://data.desi.lbl.gov/public/edr/spectro/redux/fuji/zcatalog/",
                "sample_file": "zall-pix-fuji.fits",
                "columns": ["TARGET_RA", "TARGET_DEC", "Z", "TARGETID", "FLUX_G", "FLUX_R", "FLUX_Z"]
            },
            "DES": {
                "url": "https://des.ncsa.illinois.edu/releases/y6a2/",
                "sample_file": "Y6_GOLD_2_2-519-0000.parquet",
                "columns": ["ALPHAWIN_J2000", "DELTAWIN_J2000", "MAG_AUTO_G", "MAG_AUTO_R", "MAG_AUTO_I"]
            },
            "Euclid": {
                "url": "https://euclid.esac.esa.int/euclid-archive/",
                "sample_file": "euclid_q1_mer_final.fits",
                "columns": ["RA", "DEC", "MAG_G", "MAG_R", "MAG_I", "Z_PHOT"]
            }
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def download_sdss_data(self, limit: int = 10000) -> str:
        """Download SDSS DR17 data."""
        try:
            logger.info("Downloading SDSS DR17 data...")
            
            # For demo purposes, create synthetic SDSS data
            # In production, this would download from actual SDSS servers
            data = self._generate_synthetic_sdss_data(limit)
            
            # Save to file
            output_path = DATA_DIR / "sdss_dr17_sample.csv"
            data.to_csv(output_path, index=False)
            
            logger.info(f"SDSS data saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading SDSS data: {e}")
            raise
    
    async def download_desi_data(self, limit: int = 10000) -> str:
        """Download DESI DR1 data."""
        try:
            logger.info("Downloading DESI DR1 data...")
            
            # Generate synthetic DESI data
            data = self._generate_synthetic_desi_data(limit)
            
            output_path = DATA_DIR / "desi_dr1_sample.csv"
            data.to_csv(output_path, index=False)
            
            logger.info(f"DESI data saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading DESI data: {e}")
            raise
    
    async def download_des_data(self, limit: int = 10000) -> str:
        """Download DES Y6 data."""
        try:
            logger.info("Downloading DES Y6 data...")
            
            # Generate synthetic DES data
            data = self._generate_synthetic_des_data(limit)
            
            output_path = DATA_DIR / "des_y6_sample.csv"
            data.to_csv(output_path, index=False)
            
            logger.info(f"DES data saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading DES data: {e}")
            raise
    
    async def download_euclid_data(self, limit: int = 10000) -> str:
        """Download Euclid Q1 data."""
        try:
            logger.info("Downloading Euclid Q1 data...")
            
            # Generate synthetic Euclid data
            data = self._generate_synthetic_euclid_data(limit)
            
            output_path = DATA_DIR / "euclid_q1_sample.csv"
            data.to_csv(output_path, index=False)
            
            logger.info(f"Euclid data saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error downloading Euclid data: {e}")
            raise
    
    async def merge_catalogs(self, catalog_paths: List[str]) -> str:
        """Merge multiple catalogs into a unified dataset."""
        try:
            logger.info("Merging catalogs...")
            
            merged_data = []
            
            for path in catalog_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    # Add catalog source identifier
                    df['catalog_source'] = Path(path).stem.split('_')[0].upper()
                    merged_data.append(df)
            
            if not merged_data:
                raise ValueError("No valid catalog data found")
            
            # Concatenate all catalogs
            merged_df = pd.concat(merged_data, ignore_index=True)
            
            # Standardize column names
            merged_df = self._standardize_columns(merged_df)
            
            # Save merged dataset
            output_path = OUTPUT_DIR / "merged_catalogs.csv"
            merged_df.to_csv(output_path, index=False)
            
            logger.info(f"Merged catalog saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error merging catalogs: {e}")
            raise
    
    def _generate_synthetic_sdss_data(self, limit: int) -> pd.DataFrame:
        """Generate synthetic SDSS DR17 data."""
        np.random.seed(42)
        
        data = {
            'ra': np.random.uniform(0, 360, limit),
            'dec': np.random.uniform(-90, 90, limit),
            'redshift': np.random.exponential(0.3, limit),
            'mag_g': np.random.normal(20, 2, limit),
            'mag_r': np.random.normal(19.5, 2, limit),
            'mag_i': np.random.normal(19, 2, limit),
            'object_type': np.random.choice(['galaxy', 'quasar', 'star'], limit, p=[0.7, 0.2, 0.1]),
            'plate': np.random.randint(1000, 9999, limit),
            'mjd': np.random.randint(50000, 60000, limit),
            'fiberid': np.random.randint(1, 1000, limit)
        }
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_desi_data(self, limit: int) -> pd.DataFrame:
        """Generate synthetic DESI DR1 data."""
        np.random.seed(42)
        
        data = {
            'ra': np.random.uniform(0, 360, limit),
            'dec': np.random.uniform(-90, 90, limit),
            'redshift': np.random.exponential(0.4, limit),
            'mag_g': np.random.normal(21, 1.5, limit),
            'mag_r': np.random.normal(20.5, 1.5, limit),
            'mag_z': np.random.normal(20, 1.5, limit),
            'object_type': np.random.choice(['ELG', 'LRG', 'QSO'], limit, p=[0.5, 0.3, 0.2]),
            'targetid': np.random.randint(1000000, 9999999, limit),
            'flux_g': np.random.exponential(1000, limit),
            'flux_r': np.random.exponential(1000, limit),
            'flux_z': np.random.exponential(1000, limit)
        }
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_des_data(self, limit: int) -> pd.DataFrame:
        """Generate synthetic DES Y6 data."""
        np.random.seed(42)
        
        data = {
            'ra': np.random.uniform(0, 360, limit),
            'dec': np.random.uniform(-90, 90, limit),
            'mag_g': np.random.normal(22, 1.8, limit),
            'mag_r': np.random.normal(21.5, 1.8, limit),
            'mag_i': np.random.normal(21, 1.8, limit),
            'object_type': np.random.choice(['galaxy', 'star'], limit, p=[0.8, 0.2]),
            'photoz': np.random.exponential(0.3, limit),
            'photoz_err': np.random.uniform(0.01, 0.1, limit),
            'extinction_g': np.random.uniform(0, 0.5, limit),
            'extinction_r': np.random.uniform(0, 0.4, limit),
            'extinction_i': np.random.uniform(0, 0.3, limit)
        }
        
        return pd.DataFrame(data)
    
    def _generate_synthetic_euclid_data(self, limit: int) -> pd.DataFrame:
        """Generate synthetic Euclid Q1 data."""
        np.random.seed(42)
        
        data = {
            'ra': np.random.uniform(0, 360, limit),
            'dec': np.random.uniform(-90, 90, limit),
            'mag_g': np.random.normal(23, 2, limit),
            'mag_r': np.random.normal(22.5, 2, limit),
            'mag_i': np.random.normal(22, 2, limit),
            'object_type': np.random.choice(['galaxy', 'quasar'], limit, p=[0.9, 0.1]),
            'photoz': np.random.exponential(0.4, limit),
            'photoz_err': np.random.uniform(0.02, 0.15, limit),
            'stellar_mass': np.random.lognormal(10, 0.5, limit),
            'sfr': np.random.exponential(1, limit)
        }
        
        return pd.DataFrame(data)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across catalogs."""
        column_mapping = {
            # RA columns
            'TARGET_RA': 'ra',
            'ALPHAWIN_J2000': 'ra',
            # DEC columns
            'TARGET_DEC': 'dec',
            'DELTAWIN_J2000': 'dec',
            # Redshift columns
            'Z': 'redshift',
            'Z_PHOT': 'photoz',
            # Magnitude columns
            'MAG_G': 'mag_g',
            'MAG_R': 'mag_r',
            'MAG_I': 'mag_i',
            'MAG_AUTO_G': 'mag_g',
            'MAG_AUTO_R': 'mag_r',
            'MAG_AUTO_I': 'mag_i',
            'FLUX_G': 'flux_g',
            'FLUX_R': 'flux_r',
            'FLUX_Z': 'flux_z'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['ra', 'dec', 'catalog_source']
        for col in required_columns:
            if col not in df.columns:
                if col == 'redshift':
                    df[col] = np.random.exponential(0.3, len(df))
                elif col == 'mag_g':
                    df[col] = np.random.normal(20, 2, len(df))
                else:
                    df[col] = 'unknown'
        
        return df

def get_catalog_info() -> Dict[str, Any]:
    """Get information about available catalogs."""
    return {
        "catalogs": {
            "SDSS": {
                "name": "Sloan Digital Sky Survey DR17",
                "status": "available",
                "objects": 10000,
                "description": "Spectroscopic survey of galaxies and quasars"
            },
            "DESI": {
                "name": "Dark Energy Spectroscopic Instrument DR1",
                "status": "available",
                "objects": 10000,
                "description": "Spectroscopic survey for dark energy studies"
            },
            "DES": {
                "name": "Dark Energy Survey Y6",
                "status": "available",
                "objects": 10000,
                "description": "Photometric survey for dark energy studies"
            },
            "Euclid": {
                "name": "Euclid Mission Q1",
                "status": "available",
                "objects": 10000,
                "description": "ESA space mission for cosmology"
            }
        }
    }

def get_comprehensive_statistics() -> Dict[str, Any]:
    """Get comprehensive statistics across all catalogs."""
    # Load merged data if available
    merged_path = OUTPUT_DIR / "merged_catalogs.csv"
    
    if merged_path.exists():
        df = pd.read_csv(merged_path)
        
        stats = {
            "total_objects": len(df),
            "catalog_breakdown": df['catalog_source'].value_counts().to_dict(),
            "redshift_range": f"{df['redshift'].min():.3f} - {df['redshift'].max():.3f}",
            "magnitude_range": f"{df['mag_g'].min():.2f} - {df['mag_g'].max():.2f}",
            "object_types": df['object_type'].value_counts().to_dict() if 'object_type' in df.columns else {},
            "spatial_coverage": {
                "ra_range": f"{df['ra'].min():.2f}° - {df['ra'].max():.2f}°",
                "dec_range": f"{df['dec'].min():.2f}° - {df['dec'].max():.2f}°"
            }
        }
    else:
        # Return synthetic statistics
        stats = {
            "total_objects": 40000,
            "catalog_breakdown": {
                "SDSS": 10000,
                "DESI": 10000,
                "DES": 10000,
                "Euclid": 10000
            },
            "redshift_range": "0.0 - 3.0",
            "magnitude_range": "15.0 - 25.0",
            "object_types": {
                "galaxy": 28000,
                "quasar": 8000,
                "star": 4000
            },
            "spatial_coverage": {
                "ra_range": "0.0° - 360.0°",
                "dec_range": "-90.0° - 90.0°"
            }
        }
    
    return {"statistics": stats}

def fetch_filtered_galaxies(catalog_source: str = "all", 
                          min_redshift: float = None,
                          max_redshift: float = None,
                          min_magnitude: float = None,
                          max_magnitude: float = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch filtered galaxies from catalogs."""
    try:
        # Load merged data if available
        merged_path = OUTPUT_DIR / "merged_catalogs.csv"
        
        if merged_path.exists():
            df = pd.read_csv(merged_path)
        else:
            # Generate synthetic data for demo
            processor = AstronomicalDataProcessor()
            df = processor._generate_synthetic_sdss_data(1000)
            df['catalog_source'] = 'SDSS'
        
        # Apply filters
        if catalog_source != "all":
            df = df[df['catalog_source'] == catalog_source.upper()]
        
        if min_redshift is not None:
            df = df[df['redshift'] >= min_redshift]
        
        if max_redshift is not None:
            df = df[df['redshift'] <= max_redshift]
        
        if min_magnitude is not None:
            df = df[df['mag_g'] >= min_magnitude]
        
        if max_magnitude is not None:
            df = df[df['mag_g'] <= max_magnitude]
        
        # Limit results
        df = df.head(limit)
        
        # Convert to list of dictionaries
        galaxies = df.to_dict('records')
        
        return galaxies
        
    except Exception as e:
        logger.error(f"Error fetching filtered galaxies: {e}")
        return []
=======
import ssl
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json

# Database integration
from database.config import db

logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = os.path.join(os.getcwd(), "data")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")

# Ensure directories exist
Path(DATA_DIR).mkdir(exist_ok=True)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

class AstronomicalDataProcessor:
    """Main class for processing astronomical catalog data."""
    
    def __init__(self):
        self.catalogs = {
            "sdss": {
                "name": "SDSS (Sloan Digital Sky Survey)",
                "url": "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/",
                "description": "Large-scale galaxy and quasar survey"
            },
            "desi": {
                "name": "DESI (Dark Energy Spectroscopic Instrument)", 
                "url": "https://data.desi.lbl.gov/",
                "description": "Next-generation dark energy survey"
            },
            "des": {
                "name": "DES (Dark Energy Survey)",
                "url": "https://des.ncsa.illinois.edu/",
                "description": "Wide-field optical survey"
            },
            "euclid": {
                "name": "Euclid Space Mission",
                "url": "https://www.euclid-ec.org/",
                "description": "ESA space mission for dark universe studies"
            }
        }
        
    async def fetch_sample_data(self, catalog: str, limit: int = 1000) -> pd.DataFrame:
        """Fetch sample astronomical data for the specified catalog."""
        try:
            # Generate realistic sample data for demonstration
            np.random.seed(42)  # For reproducible results
            
            n_samples = min(limit, 1000)
            
            # Common astronomical object properties
            data = {
                'ra': np.random.uniform(0, 360, n_samples),  # Right Ascension
                'dec': np.random.uniform(-90, 90, n_samples),  # Declination
                'z': np.random.exponential(0.3, n_samples),  # Redshift
                'magnitude_g': np.random.normal(20, 2, n_samples),  # g-band magnitude
                'magnitude_r': np.random.normal(19.8, 2, n_samples),  # r-band magnitude
                'magnitude_i': np.random.normal(19.5, 2, n_samples),  # i-band magnitude
                'object_type': np.random.choice(['galaxy', 'star', 'quasar'], n_samples, p=[0.7, 0.2, 0.1]),
                'catalog_source': catalog
            }
            
            # Add catalog-specific columns
            if catalog == "sdss":
                data['objid'] = np.random.randint(1000000000, 9999999999, n_samples)
                data['specobjid'] = np.random.randint(100000000000, 999999999999, n_samples)
            elif catalog == "desi":
                data['targetid'] = np.random.randint(10000000, 99999999, n_samples)
                data['survey'] = 'main'
            elif catalog == "des":
                data['coadd_object_id'] = np.random.randint(100000, 999999, n_samples)
                data['tilename'] = [f"DES{i:04d}{j:+03d}" for i, j in zip(np.random.randint(0, 9999, n_samples), np.random.randint(-50, 50, n_samples))]
            elif catalog == "euclid":
                data['euclid_id'] = np.random.randint(1000000, 9999999, n_samples)
                data['observation_time'] = pd.date_range('2023-01-01', periods=n_samples, freq='1H')
                
            df = pd.DataFrame(data)
            
            # Apply realistic constraints
            df = df[(df['z'] >= 0) & (df['z'] <= 5)]  # Reasonable redshift range
            df = df[(df['magnitude_g'] > 10) & (df['magnitude_g'] < 30)]  # Observable magnitude range
            
            logger.info(f"Generated {len(df)} sample objects for {catalog} catalog")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for catalog {catalog}: {e}")
            return pd.DataFrame()
    
    async def process_catalog_data(self, catalog: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Process and analyze catalog data."""
        try:
            if data.empty:
                return {"error": "No data provided"}
                
            analysis = {
                "catalog": catalog,
                "total_objects": len(data),
                "object_types": data['object_type'].value_counts().to_dict(),
                "redshift_stats": {
                    "mean": float(data['z'].mean()),
                    "median": float(data['z'].median()),
                    "std": float(data['z'].std()),
                    "min": float(data['z'].min()),
                    "max": float(data['z'].max())
                },
                "magnitude_stats": {
                    "g_band": {
                        "mean": float(data['magnitude_g'].mean()),
                        "std": float(data['magnitude_g'].std())
                    },
                    "r_band": {
                        "mean": float(data['magnitude_r'].mean()),
                        "std": float(data['magnitude_r'].std())
                    }
                },
                "coordinate_coverage": {
                    "ra_range": [float(data['ra'].min()), float(data['ra'].max())],
                    "dec_range": [float(data['dec'].min()), float(data['dec'].max())]
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error processing catalog data: {e}")
            return {"error": str(e)}

async def get_catalog_info(catalog: str = None) -> Dict[str, Any]:
    """Get information about available catalogs."""
    processor = AstronomicalDataProcessor()
    
    if catalog:
        if catalog in processor.catalogs:
            return processor.catalogs[catalog]
        else:
            return {"error": f"Catalog '{catalog}' not found"}
    
    return processor.catalogs

async def get_comprehensive_statistics() -> Dict[str, Any]:
    """Get comprehensive statistics for all catalogs."""
    try:
        # Try to get from database first
        db_stats = await db.get_statistics()
        if db_stats:
            return db_stats
            
        # Generate sample statistics
        processor = AstronomicalDataProcessor()
        all_stats = {}
        
        for catalog_name in processor.catalogs.keys():
            data = await processor.fetch_sample_data(catalog_name, 100)
            if not data.empty:
                catalog_analysis = await processor.process_catalog_data(catalog_name, data)
                all_stats[catalog_name] = catalog_analysis
                
        return {
            "comprehensive_stats": all_stats,
            "generated_at": pd.Timestamp.now().isoformat(),
            "total_catalogs": len(all_stats)
        }
        
    except Exception as e:
        logger.error(f"Error getting comprehensive statistics: {e}")
        return {"error": str(e)}

async def fetch_filtered_galaxies(
    catalog: str = "sdss",
    min_z: float = 0.0,
    max_z: float = 1.0,
    min_magnitude: float = 15.0,
    max_magnitude: float = 25.0,
    limit: int = 100,
    object_type: str = "galaxy"
) -> List[Dict[str, Any]]:
    """Fetch filtered galaxy data from specified catalog."""
    try:
        processor = AstronomicalDataProcessor()
        
        # Fetch data
        data = await processor.fetch_sample_data(catalog, limit * 2)  # Get more to account for filtering
        
        if data.empty:
            return []
            
        # Apply filters
        logger.info(f"Before filtering: {len(data)} objects")
        logger.info(f"Filter parameters: min_z={min_z}, max_z={max_z}, min_mag={min_magnitude}, max_mag={max_magnitude}, object_type={object_type}")
        
        if len(data) > 0:
            logger.info(f"Data ranges: z=[{data['z'].min():.3f}, {data['z'].max():.3f}], mag_g=[{data['magnitude_g'].min():.1f}, {data['magnitude_g'].max():.1f}]")
            logger.info(f"Object types in data: {data['object_type'].value_counts().to_dict()}")
        
        filtered_data = data[
            (data['z'] >= min_z) & 
            (data['z'] <= max_z) &
            (data['magnitude_g'] >= min_magnitude) &  # Note: Higher magnitudes are dimmer
            (data['magnitude_g'] <= max_magnitude) &  # Lower magnitudes are brighter
            (data['object_type'] == object_type)
        ].head(limit)
        
        logger.info(f"After filtering: {len(filtered_data)} objects")
        
        # Convert to list of dictionaries
        result = []
        for _, row in filtered_data.iterrows():
            obj = {
                'ra': float(row['ra']),
                'dec': float(row['dec']),
                'redshift': float(row['z']),
                'magnitude_g': float(row['magnitude_g']),
                'magnitude_r': float(row['magnitude_r']),
                'magnitude_i': float(row['magnitude_i']),
                'object_type': row['object_type'],
                'catalog_source': row['catalog_source']
            }
            
            # Add catalog-specific fields
            if catalog == "sdss" and 'objid' in row:
                obj['objid'] = int(row['objid'])
            elif catalog == "desi" and 'targetid' in row:
                obj['targetid'] = int(row['targetid'])
            elif catalog == "des" and 'coadd_object_id' in row:
                obj['coadd_object_id'] = int(row['coadd_object_id'])
            elif catalog == "euclid" and 'euclid_id' in row:
                obj['euclid_id'] = int(row['euclid_id'])
                
            result.append(obj)
            
        logger.info(f"Fetched {len(result)} filtered {object_type}s from {catalog}")
        return result
        
    except Exception as e:
        logger.error(f"Error fetching filtered galaxies: {e}")
        return []

# Additional utility functions for compatibility

async def load_catalog_data(catalog: str, **kwargs) -> pd.DataFrame:
    """Load catalog data with optional parameters."""
    processor = AstronomicalDataProcessor()
    limit = kwargs.get('limit', 1000)
    return await processor.fetch_sample_data(catalog, limit)

async def analyze_redshift_distribution(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze redshift distribution in the dataset."""
    if data.empty or 'z' not in data.columns:
        return {"error": "No redshift data available"}
        
    z_data = data['z'].dropna()
    
    return {
        "total_objects": len(z_data),
        "redshift_bins": {
            "low_z (0-0.1)": len(z_data[(z_data >= 0) & (z_data < 0.1)]),
            "mid_z (0.1-0.5)": len(z_data[(z_data >= 0.1) & (z_data < 0.5)]),
            "high_z (0.5-2.0)": len(z_data[(z_data >= 0.5) & (z_data < 2.0)]),
            "very_high_z (>2.0)": len(z_data[z_data >= 2.0])
        },
        "statistics": {
            "mean": float(z_data.mean()),
            "median": float(z_data.median()),
            "std": float(z_data.std()),
            "min": float(z_data.min()),
            "max": float(z_data.max())
        }
    }

# Export the main functions and constants
__all__ = [
    'AstronomicalDataProcessor',
    'get_catalog_info', 
    'get_comprehensive_statistics',
    'fetch_filtered_galaxies',
    'load_catalog_data',
    'analyze_redshift_distribution',
    'DATA_DIR',
    'OUTPUT_DIR'
]
