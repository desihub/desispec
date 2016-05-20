--
-- Schema for DESI pipeline file-tracking database.
-- Note: the SQL flavor is tuned to SQLite.
--
CREATE TABLE brick (
    brickid INTEGER PRIMARY KEY,
    brickname TEXT UNIQUE NOT NULL,
    brickq INTEGER NOT NULL,
    brickrow INTEGER NOT NULL,
    brickcol INTEGER NOT NULL,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    ra1 REAL NOT NULL,
    ra2 REAL NOT NULL,
    dec1 REAL NOT NULL,
    dec2 REAL NOT NULL,
    area REAL NOT NULL
);
--
--
--
CREATE TABLE filetype (
    type TEXT PRIMARY KEY
);
--
--
--
CREATE TABLE file (
    id TEXT PRIMARY KEY, -- Checksum of the file.  SHA1 preferred.
    filename TEXT NOT NULL,
    directory TEXT NOT NULL,
    prodname TEXT NOT NULL,
    filetype TEXT NOT NULL, -- Foreign key on filetype
    FOREIGN KEY (filetype) REFERENCES filetype (type)
);
--
-- Both fileid and requires are primary keys in file.
-- 'requires' == 'needs this file'
--
CREATE TABLE filedependency (
    fileid TEXT NOT NULL,
    requires TEXT NOT NULL, -- Primary key on the two columns (fileid, requires)
    PRIMARY KEY (fileid, requires),
    FOREIGN KEY (fileid) REFERENCES file (id),
    FOREIGN KEY (requires) REFERENCES file (id)
);
--
-- JOIN table
--
CREATE TABLE file2brick (
    fileid TEXT NOT NULL,
    brickid INTEGER NOT NULL,
    PRIMARY KEY (fileid, brickid),
    FOREIGN KEY (fileid) REFERENCES file (id),
    FOREIGN KEY (brickid) REFERENCES brick (brickid)
);
--
--
--
CREATE TABLE night (
    night INTEGER PRIMARY KEY -- e.g. 20150510
);
--
--
--
CREATE TABLE exposureflavor (
    flavor TEXT PRIMARY KEY
);
--
--
--
CREATE TABLE exposure (
    expid INTEGER PRIMARY KEY,
    night INTEGER NOT NULL, -- foreign key on night
    flavor TEXT NOT NULL, -- arc, flat, science, etc. might want a separate table?
    telra REAL NOT NULL,
    teldec REAL NOT NULL,
    tileid INTEGER NOT NULL DEFAULT -1, -- it is possible for the telescope to not point at a tile.
    exptime REAL NOT NULL,
    dateobs TIMESTAMP NOT NULL, -- text or integer are also possible here.  TIMESTAMP allows automatic conversion to Python datetime objects.
    alt REAL NOT NULL,
    az REAL NOT NULL,
    FOREIGN KEY (night) REFERENCES night (night),
    FOREIGN KEY (flavor) REFERENCES exposureflavor (flavor)
);
--
-- JOIN table
--
CREATE TABLE file2exposure (
    fileid TEXT NOT NULL,
    expid INTEGER NOT NULL,
    PRIMARY KEY (fileid, expid),
    FOREIGN KEY (fileid) REFERENCES file (id),
    FOREIGN KEY (expid) REFERENCES exposure (expid)
);
--
-- JOIN table
--
CREATE TABLE exposure2brick (
    expid INTEGER NOT NULL,
    brickid INTEGER NOT NULL,
    PRIMARY KEY (expid, brickid),
    FOREIGN KEY (expid) REFERENCES exposure (expid),
    FOREIGN KEY (brickid) REFERENCES brick (brickid)
);
