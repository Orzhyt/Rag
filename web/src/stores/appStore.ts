import { create } from 'zustand';

const STORAGE_KEY = 'rag-app-state';

function loadPersisted(): Pick<AppState, 'selectedDatabase' | 'selectedCollections'> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch { /* ignore */ }
  return { selectedDatabase: null, selectedCollections: [] };
}

function persist(state: Pick<AppState, 'selectedDatabase' | 'selectedCollections'>) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch { /* ignore */ }
}

interface AppState {
  selectedDatabase: string | null;
  selectedCollections: string[];
  sidebarCollapsed: boolean;

  setSelectedDatabase: (db: string | null) => void;
  setSelectedCollections: (cols: string[]) => void;
  toggleSidebar: () => void;
}

const initial = loadPersisted();

export const useAppStore = create<AppState>((set, get) => ({
  selectedDatabase: initial.selectedDatabase,
  selectedCollections: initial.selectedCollections,
  sidebarCollapsed: false,

  setSelectedDatabase: (db) => {
    set({ selectedDatabase: db });
    persist({ selectedDatabase: db, selectedCollections: get().selectedCollections });
  },
  setSelectedCollections: (cols) => {
    set({ selectedCollections: cols });
    persist({ selectedDatabase: get().selectedDatabase, selectedCollections: cols });
  },
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
}));
