--
-- Schema for DESI pipeline raw data tracking database.
-- Note: the SQL flavor is tuned to SQLite.
--
-- This schema might be used for both simulations and real data.  For
-- simple simulations, the tile table will be filled, while for realistic
-- simulations and real data, the frame table will be filled.
--
CREATE TABLE tile (
    tileid INTEGER PRIMARY KEY,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    pass INTEGER NOT NULL,
    in_desi INTEGER NOT NULL
);
--
--
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
CREATE TABLE night (
    night INTEGER PRIMARY KEY -- e.g. 20150510
);
--
--
--
CREATE TABLE exposureflavor (
    flavor TEXT PRIMARY KEY -- arc, flat, science, etc.
);
INSERT INTO exposureflavor (flavor) VALUES ('science');
INSERT INTO exposureflavor (flavor) VALUES ('arc');
INSERT INTO exposureflavor (flavor) VALUES ('flat');
--
--
--
CREATE TABLE frame (
    frameid TEXT PRIMARY KEY, -- e.g. b0-00012345
    band TEXT NOT NULL, -- b, r, z, might be called 'channel' or 'arm'
    spectrograph INTEGER NOT NULL, -- 0, 1, 2, ...
    expid INTEGER NOT NULL, -- exposure number
    night INTEGER NOT NULL, -- foreign key on night
    flavor TEXT NOT NULL, -- foreign key on exposureflavor, might be called 'obstype'
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
-- If frame table becomes two big, create an exposure table to contain
-- the data that is common to all frames in an exposure.
--
-- JOIN table
--
CREATE TABLE frame2brick (
    frameid TEXT NOT NULL,
    brickid INTEGER NOT NULL,
    PRIMARY KEY (frameid, brickid),
    FOREIGN KEY (frameid) REFERENCES frame (frameid),
    FOREIGN KEY (brickid) REFERENCES brick (brickid)
);
--
-- JOIN table
--
CREATE TABLE tile2brick (
    tileid INTEGER NOT NULL,
    petalid INTEGER NOT NULL,
    brickid INTEGER NOT NULL,
    PRIMARY KEY (tileid, petalid, brickid),
    FOREIGN KEY (tileid) REFERENCES tile (tileid),
    FOREIGN KEY (brickid) REFERENCES brick (brickid)
);
--
-- Status
--
CREATE TABLE status (
    status TEXT PRIMARY KEY  -- not processed, failed, succeeded
);
INSERT INTO status (status) VALUES ('not processed');
INSERT INTO status (status) VALUES ('failed');
INSERT INTO status (status) VALUES ('succeeded');
--
--
--
CREATE TABLE framestatus (
    frameid TEXT NOT NULL,
    status TEXT NOT NULL,
    stamp TIMESTAMP NOT NULL,
    FOREIGN KEY (status) REFERENCES status (status)
);
--
--
--
CREATE TABLE brickstatus (
    brickid INTEGER NOT NULL,
    status TEXT NOT NULL,
    stamp TIMESTAMP NOT NULL,
    FOREIGN KEY (status) REFERENCES status (status)
);
--
-- Index
--
CREATE INDEX brick_ra1_idx ON brick (ra1);
CREATE INDEX brick_ra2_idx ON brick (ra2);
CREATE INDEX brick_dec1_idx ON brick (dec1);
CREATE INDEX brick_dec2_idx ON brick (dec2);
