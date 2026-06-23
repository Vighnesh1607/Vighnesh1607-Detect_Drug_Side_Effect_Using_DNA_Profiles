"""
Genetic Score Calculator (calculation module)
-------------------------------------------------
Pure calculation logic, no Streamlit / no API / no external files.
Import calculate_genetic_score() from this module wherever you need
to compute a person'''s genetic_score from their gene-level CSV.

Reference values (per-gene caps + min/max raw-score range) were
computed once from a 2504-person reference population and are
hardcoded below as plain Python constants.
"""

import csv
import io


# ---------------------------------------------------------------------------
# FIXED REFERENCE VALUES (calculated once from a 2504-person reference dataset)
# Do not edit unless you intentionally want to refit against a new reference
# population (these define how every score compares to the rest).
# ---------------------------------------------------------------------------

GENE_NAMES = [
    "LINC00266-3", "CICP18", "RP3-416J7.1", "RP3-416J7.4",
    "RP3-416J7.5", "DUSP22", "IRF4", "EXOC2",
    "RP11-532F6.3", "RP11-532F6.4", "RP11-532F6.5", "RP5-1077H22.2",
    "RP5-1077H22.1", "RP5-856G1.1", "AL033381.1", "snoU13",
    "FOXQ1", "RP4-668J24.2", "FOXF2", "RN7SL352P",
    "RP11-157J24.1", "RP11-157J24.2", "FOXC1", "GMDS",
    "GMDS-AS1", "C6orf195", "RP11-145H9.3", "MYLK4",
    "WRNIP1", "SERPINB1", "MIR4645", "RP11-420G6.4",
    "SERPINB9", "RP1-90J20.10", "RP1-90J20.2", "SERPINB6",
    "LINC01011", "NQO2", "RP1-90J20.12", "HTATSF1P2",
    "RP1-90J20.11", "FAM136BP", "RP1-40E16.2", "RIPK1",
    "BPHL", "TUBB2A", "TUBB2BP1", "RP1-40E16.9",
    "AC093627.7", "AC093627.8", "AC093627.9", "AC093627.10",
    "RP11-90P13.1", "AC093627.11", "AC093627.12", "FAM20C",
    "AC187652.1", "WI2-2373I1.2", "AC226118.1", "AC147651.1",
    "PDGFA", "AC147651.3", "PRKAR1B", "HEATR2",
    "SUN1", "ADAP1", "COX19", "CYP2W1",
    "C7orf50", "ZFAND2A", "AC091729.9", "UNCX",
    "AC073094.4", "MICALL2", "AC102953.4", "RP11-1246C19.1",
    "INTS1", "AC102953.6", "MAFK", "TMEM184A",
    "PSMG3", "PSMG3-AS1", "AC074389.5", "ELFN1",
    "MAD1L1", "FTSJ2", "NUDT1", "SNX8",
    "EIF3B", "AC004840.8", "RP11-631M21.6", "TUBB8",
    "IL9RP2", "ZMYND11", "DIP2C", "RP11-164C1.2",
    "LARP4B", "RP11-363N22.2", "AL359878.1", "GTPBP4",
    "IDI2", "IDI2-AS1", "IDI1", "WDR37",
    "LINC00200", "ADARB2", "LINC00700", "RP11-69C17.2",
    "RNU6-889P", "RP11-69C17.3", "RP11-69C17.4", "RNU6-576P",
    "LINC00701", "RP11-446F3.2", "RP11-526P5.1", "RP11-526P5.2",
    "RP11-598F7.1", "ABC7-42389800N19.1", "DDX11L8", "AC026369.1",
    "FAM138D", "IQSEC3", "RP11-598F7.6", "SLC6A12",
    "SLC6A13", "RP11-283I3.6", "KDM5A", "CCDC77",
    "B4GALNT3", "NINJ2", "RP11-218M22.2", "WNK1",
    "RAD52", "RP11-359B12.2", "ERC1", "LINC00942",
    "WNT5B", "ADIPOR2", "CACNA2D4", "RP5-1096D14.3",
    "LINC00940", "DCP1B", "CACNA1C", "RP11-885B4.1",
    "RP4-816N1.1", "CBX3P4", "FKBP4", "RP4-816N1.6",
    "ITFG2", "FOXM1", "RHNO1", "TULP3",
    "TEAD4", "RP11-253E3.3", "RP11-253E3.1", "TSPAN9",
    "DDX11L10", "WASH4P", "Z84812.4", "IL9RP3",
    "POLR3K", "SNRNP25", "RHBDF1", "MPG",
    "NPRL3", "HBZ", "HBM", "Z84721.4",
    "HBA2", "HBA1", "Y_RNA", "HBQ1",
    "LA16c-OS12.2", "LUC7L", "Z69890.1", "ITFG3",
    "RGS11", "ARHGDIG", "PDIA2", "AXIN1",
    "MRPL28", "TMEM8A", "Z97634.3", "NME4",
    "DECR2", "RAB11FIP3", "CAPN15", "C16orf11",
    "PIGQ", "RAB40C", "WFIKKN1", "C16orf13",
    "AL022341.1", "FAM195A", "AL022341.3", "WDR90",
    "RHOT2", "RHBDL1", "LA16c-313D11.9", "STUB1",
    "JMJD8", "WDR24", "LA16c-313D11.12", "FBXL16",
    "LA16c-380A1.1", "METRN", "FAM173A", "CCDC78",
    "HAGHL", "NARFL", "MSLN", "MSLNL",
    "RPUSD1", "CHTF18", "PRR25", "LMF1",
    "RP11-161M6.2", "SOX8", "RP11-161M6.3", "SSTR5-AS1",
    "SSTR5", "C1QTNF8", "LA16c-349E11.1", "LA16c-381G6.1",
    "CACNA1H", "TPSG1", "TPSB2", "TPSAB1",
    "TPSD1", "PRSS29P", "RP11-616M22.10", "RP11-616M22.9",
    "TPSP2", "RP11-616M22.7", "UBE2I", "RPS20P2",
    "BAIAP3", "TSR3", "GNPTG", "UNKL",
    "C16orf91", "CCDC154", "CLCN7", "RPS3AP2",
    "PTX4", "TELO2", "IFT140", "CRAMP1L",
    "LA16c-431H6.6", "HN1L", "MAPK8IP3", "NME3",
    "MRPS34", "EME2", "SPSB3", "IGFALS",
    "HAGH", "FAHD1", "MEIOB", "XX-DJ76P10__A.2",
    "RN7SL367P", "HS3ST6", "MSRB1", "RPL3L",
    "NDUFB10", "RPS2", "SNHG9", "RNF151",
    "TBL3", "NOXO1", "GFER", "AC005606.14",
    "SYNGR3", "AC005606.15", "ZNF598", "NPW",
    "SLC9A3R2", "NTHL1", "TSC2", "PKD1",
    "RAB26", "RP11-304L19.5", "TRAF7", "CASKIN1",
    "MLST8", "BRICD5", "RP11-304L19.8", "PGP",
    "E4F1", "RP11-304L19.12", "DNASE1L2", "ECI1",
    "RNPS1", "MIR940", "ABCA3", "ABCA17P",
]

GENE_CAPS = [
    1.0, 0.0, 1.0, 16.0, 10.0, 155.0, 62.0, 659.0,
    12.0, 155.0, 53.84999999999991, 16.0, 15.0, 11.0, 106.84999999999991, 1.0,
    14.0, 5.0, 16.0, 4.0, 8.0, 10.0, 6.0, 1427.85,
    587.0, 35.0, 14.0, 436.0, 67.0, 43.0, 0.0, 142.0,
    42.0, 4.0, 4.0, 108.0, 14.0, 146.0, 42.0, 5.0,
    32.0, 0.0, 4.0, 179.0, 136.8499999999999, 10.0, 16.0, 67.0,
    10.0, 35.0, 40.0, 12.0, 12.0, 8.0, 63.0, 326.0,
    9.0, 25.0, 28.0, 42.0, 117.0, 25.0, 557.0, 296.0,
    310.0, 226.0, 84.0, 22.0, 592.0, 51.0, 97.0, 10.0,
    10.0, 129.0, 6.0, 12.0, 209.0, 12.0, 42.0, 93.0,
    12.0, 47.0, 4.0, 125.0, 1488.0, 18.84999999999991, 45.0, 310.0,
    100.84999999999991, 23.0, 6.0, 85.0, 18.0, 242.0, 1255.85, 7.0,
    237.0, 25.0, 58.0, 136.0, 25.0, 82.84999999999991, 18.0, 229.0,
    28.0, 2261.7, 39.0, 38.0, 0.0, 14.0, 70.0, 0.0,
    87.0, 9.0, 8.0, 356.8499999999999, 4.0, 40.0, 2.0, 3.0,
    7.0, 352.0, 16.0, 99.0, 138.0, 8.0, 301.0, 157.0,
    422.8499999999999, 431.0, 38.0, 379.0, 296.0, 1.0, 1454.85, 54.0,
    410.0, 282.0, 419.0, 22.0, 39.0, 110.0, 1573.0, 163.0,
    4.0, 9.0, 14.0, 23.0, 163.0, 55.0, 63.0, 384.0,
    340.0, 20.0, 4.0, 385.0, 10.0, 12.0, 30.0, 20.0,
    43.0, 18.0, 94.0, 54.0, 200.0, 8.0, 65.0, 3.0,
    3.0, 1.0, 0.0, 3.0, 9.0, 64.0, 2.0, 136.0,
    39.0, 60.0, 12.0, 250.0, 18.0, 131.0, 44.0, 70.0,
    7.0, 275.0, 91.0, 19.0, 111.0, 146.0, 31.0, 2.0,
    1.0, 14.0, 4.0, 116.0, 27.0, 1.0, 2.0, 5.0,
    1.0, 16.0, 13.0, 38.0, 2.0, 21.0, 15.0, 17.0,
    38.0, 18.0, 54.0, 69.0, 14.0, 72.0, 51.0, 791.0,
    1.0, 27.0, 39.0, 71.0, 14.0, 28.0, 18.0, 9.0,
    229.0, 22.0, 41.0, 12.0, 13.0, 20.0, 20.0, 2.0,
    11.0, 11.0, 170.0, 8.0, 62.0, 6.0, 30.0, 181.0,
    32.0, 61.0, 163.0, 3.0, 23.0, 54.0, 333.0, 134.8499999999999,
    16.0, 18.0, 158.0, 5.0, 4.0, 28.0, 62.0, 21.0,
    367.0, 135.0, 505.0, 8.0, 2.0, 49.0, 10.0, 78.0,
    26.0, 27.0, 4.0, 8.0, 46.0, 12.0, 19.0, 18.0,
    6.0, 6.0, 65.0, 44.0, 42.0, 8.0, 87.84999999999991, 139.8499999999999,
    31.0, 2.0, 26.0, 25.0, 12.0, 18.0, 5.0, 6.0,
    25.0, 2.0, 5.0, 35.0, 28.0, 6.0, 142.0, 225.8499999999999,
]

REF_RAW_MIN = 20018.85
REF_RAW_MAX = 26894.55
SCALED_LOW = 100
SCALED_HIGH = 1000


def compute_genetic_score(values):
    """Cap each gene value at its reference cap, sum, then scale to [SCALED_LOW, SCALED_HIGH].

    Args:
        values: list of int/float gene values, in the SAME order as GENE_NAMES.

    Returns:
        (raw_score, scaled_score) tuple.
    """
    capped_values = [min(v, cap) for v, cap in zip(values, GENE_CAPS)]
    raw_score = sum(capped_values)

    if REF_RAW_MAX == REF_RAW_MIN:
        scaled = (SCALED_LOW + SCALED_HIGH) / 2
    else:
        scaled = SCALED_LOW + (raw_score - REF_RAW_MIN) * (SCALED_HIGH - SCALED_LOW) / (REF_RAW_MAX - REF_RAW_MIN)

    scaled = max(SCALED_LOW, min(SCALED_HIGH, scaled))
    return raw_score, round(scaled, 2)


def parse_gene_csv(file_bytes):
    """Parse gene CSV bytes into rows of (person_id, values list).

    Args:
        file_bytes: raw bytes of an uploaded/read CSV file
                    (header: Person + 296 gene columns).

    Returns:
        (rows, error_message) tuple. error_message is None on success,
        rows is None on failure.
    """
    try:
        text = file_bytes.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        header = next(reader)
    except Exception:
        return None, "Could not read the file. Please upload a valid CSV."

    gene_cols = header[1:]
    if gene_cols != GENE_NAMES:
        return None, (
            "Column mismatch: this file'''s gene columns don'''t match the expected "
            "reference gene list. Make sure this is the gene-level CSV produced by the "
            "pipeline (Person + 296 gene columns)."
        )

    rows = []
    for row in reader:
        if not row:
            continue
        person_id = row[0]
        try:
            values = [int(v) for v in row[1:]]
        except ValueError:
            return None, f"Non-numeric value found in row for person {person_id}."
        if len(values) != len(GENE_NAMES):
            return None, f"Row for person {person_id} has {len(values)} values, expected {len(GENE_NAMES)}."
        rows.append((person_id, values))

    if not rows:
        return None, "CSV has a header but no data rows."

    return rows, None


def calculate_genetic_score(file_bytes):
    """
    Main entry point: takes raw CSV file bytes (Person + 296 gene columns,
    one or more person rows) and returns the computed score(s).

    Args:
        file_bytes: raw bytes of the uploaded/read gene CSV file.

    Returns:
        (results, error_message) tuple.
        - On success: results is a list of dicts like
            [{"person_id": "1", "raw_score_capped": 22038.0, "genetic_score": 364.3}, ...]
          and error_message is None.
        - On failure: results is None and error_message explains what went wrong.
    """
    rows, error = parse_gene_csv(file_bytes)
    if error:
        return None, error

    results = []
    for person_id, values in rows:
        raw_score, scaled_score = compute_genetic_score(values)
        results.append({
            "person_id": person_id,
            "raw_score_capped": raw_score,
            "genetic_score": scaled_score,
        })

    return results, None
