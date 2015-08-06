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
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL
);
--
--
--
CREATE TABLE file (
    id INTEGER PRIMARY KEY,
    filename TEXT NOT NULL,
    directory TEXT NOT NULL,
    prodname TEXT NOT NULL,
    filetype INTEGER NOT NULL, -- Foreign key on filetype
    FOREIGN KEY (filetype) REFERENCES filetype (id)
);
--
-- Both fileid and requires are primary keys in file.
-- 'requires' == 'needs this file'
--
CREATE TABLE filedependcy (
    fileid INTEGER NOT NULL,
    requires INTEGER NOT NULL, -- Primary key on the two columns (fileid, requires)
    PRIMARY KEY (fileid, requires),
    FOREIGN KEY (fileid) REFERENCES file (id),
    FOREIGN KEY (requires) REFERENCES file (id)
);
--
-- JOIN table
--
CREATE TABLE file2brick (
    fileid INTEGER NOT NULL,
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
    id INTEGER PRIMARY KEY,
    flavor TEXT UNIQUE NOT NULL
);
--
--
--
CREATE TABLE exposure (
    expid INTEGER PRIMARY KEY,
    night INTEGER NOT NULL, -- foreign key on night
    flavor INTEGER NOT NULL, -- arc, flat, science, etc. might want a separate table?
    telra REAL NOT NULL,
    teldec REAL NOT NULL,
    tileid INTEGER, -- it is possible for the telescope to not point at a tile.
    exptime REAL NOT NULL,
    dateobs DATETIME, -- text or integer are also possible here.
    alt REAL NOT NULL,
    az REAL NOT NULL,
    FOREIGN KEY (night) REFERENCES night (night),
    FOREIGN KEY (flavor) REFERENCES exposureflavor (id)
);
--
-- JOIN table
--
CREATE TABLE file2exposure (
    fileid INTEGER NOT NULL,
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
